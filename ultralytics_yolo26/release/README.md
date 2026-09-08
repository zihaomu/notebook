# YOLO26 workshop release lock

`current.env` defines the two-image release consumed by the launcher. Runtime
references are immutable OCI index digests; human-readable tags are recorded for
discovery only.

Release lock files are host-side deployment metadata. They are not copied into the
pipeline image and do not participate in the workshop bundle SHA. The image carries
its source commit, release ID, companion digest, bundle SHA, and model-set SHA in OCI
labels and environment variables; the post-push lock then binds those identities to
the final pipeline digest without creating a digest self-reference.

The current entry is a bootstrap lock for the already published and GPU-verified
2026-09-04 pipeline image plus its existing llama.cpp companion. The old pipeline
image predates release-ID and companion-digest OCI labels, so the validator checks
its source commit, bundle SHA, and model-set SHA. The next clean rebuild must carry
all release labels and replace this bootstrap lock with a new release record.

Published lock files are immutable. For a new release:

1. Commit all source changes and build from a tracked-clean worktree.
2. Build and run static plus W7900D GPU gates against that commit.
3. Obtain explicit approval for both exact release-only ACR tags before tagging or pushing.
4. Push without overwriting an existing tag and record both OCI index and linux/amd64 manifest digests.
5. Add an archived lock named after the release ID, then update `current.env` in a separate release-lock commit.

Environment variables may override the lock for local development. Such overrides
are not a published release and must be reported as development mode.
