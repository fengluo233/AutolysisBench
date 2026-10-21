"""Download the published liver patches into the original four folder names."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

SPLITS = ("trainA", "trainB", "testA", "testB")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_shard(archive_path, output_dir, entries):
    """Extract only manifest-listed PNGs; preserve and check their original bytes."""
    output_dir = Path(output_dir)
    seen = set()
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).name != member.name:
                raise ValueError("Unexpected archive member")
            if member.name.endswith(".json"):
                continue
            if not member.name.endswith(".png"):
                raise ValueError("Unexpected archive extension")
            sample_id = member.name[:-4]
            if sample_id not in entries or sample_id in seen:
                raise ValueError("Unexpected or repeated image in archive")
            row = entries[sample_id]
            if row["split"] not in SPLITS:
                raise ValueError("Unknown split in manifest")
            stream = archive.extractfile(member)
            payload = stream.read()
            if len(payload) != row["png_bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise ValueError("PNG checksum mismatch: " + sample_id)
            destination = output_dir / row["split"] / member.name
            with destination.open("xb") as target:
                target.write(payload)
            seen.add(sample_id)
    if seen != set(entries):
        raise ValueError("Archive does not contain all manifest-listed images")
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset ID")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--splits", choices=SPLITS, nargs="+", default=list(SPLITS))
    parser.add_argument("--revision", default="main", help="Dataset branch, tag or commit")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The output directory must not already exist")
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    # Resolve the revision once so a concurrent repository update cannot mix versions.
    revision = HfApi().dataset_info(args.repo_id, revision=args.revision).sha

    def download(path):
        return Path(hf_hub_download(args.repo_id, path, repo_type="dataset", revision=revision))

    try:
        summary = json.loads(download("release_summary.json").read_text())
        rows = [json.loads(line) for line in download("release_manifest.jsonl").read_text().splitlines()]
    except GatedRepoError:
        parser.exit(2, "Request access at https://huggingface.co/datasets/{} (automatic approval), "
                    "then run hf auth login with the same account and retry.\n".format(args.repo_id))
    entries_by_shard = {}
    for row in rows:
        if row["split"] in args.splits:
            entries_by_shard.setdefault(row["shard"], {})[row["sample_id"]] = row
    args.output.mkdir(parents=True)
    for split in set(args.splits):
        (args.output / split).mkdir()
    counts = {split: 0 for split in set(args.splits)}
    for archive in summary["archives"]:
        path = archive["path"]
        if path not in entries_by_shard:
            continue
        cached = download(path)
        if cached.stat().st_size != archive["bytes"] or file_sha256(cached) != archive["sha256"]:
            raise ValueError("Archive checksum mismatch: " + path)
        entries = entries_by_shard[path]
        count = extract_shard(cached, args.output, entries)
        split = next(iter(entries.values()))["split"]
        counts[split] += count
        print("{}: {} images extracted".format(path, count), flush=True)
    for split, count in counts.items():
        if count != summary["counts"][split]:
            raise ValueError("Unexpected final image count for " + split)
    (args.output / "download_record.json").write_text(json.dumps({
        "repo_id": args.repo_id, "revision": revision, "counts": counts,
    }, indent=2) + "\n")
    print("Verified PNG folders are ready at " + str(args.output))


if __name__ == "__main__":
    main()
