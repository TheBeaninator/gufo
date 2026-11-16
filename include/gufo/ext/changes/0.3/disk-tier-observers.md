# Disk-tier observers

Additive: a provider whose cache payloads refer to bytes it keeps elsewhere can
follow the disk tier's contents.

- `DiskStore::SetRemovalObserver` (`src/cache/disk.hpp`) runs after each chunk
  or private file is unlinked, by retirement or reclamation, under the store's
  locks.
- `ComponentCacheResources` (`src/cache/component_resources.hpp`) gains the
  optional `disk_opened`, `disk_published` and `disk_removed` callbacks. The
  component text cache calls `disk_opened` after startup discovery,
  `disk_published` once a checkpoint is durable and before its persistence
  sources are released, and installs `disk_removed` as the store's removal
  observer.

Migration: none; unset callbacks keep the previous behavior.
