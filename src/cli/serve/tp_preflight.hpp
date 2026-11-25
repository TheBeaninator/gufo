#ifndef GUFO_CLI_SERVE_TP_PREFLIGHT_HPP_
#define GUFO_CLI_SERVE_TP_PREFLIGHT_HPP_

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gufo::server {

/// Host memory a TP2 rank registers with the RDMA device when it falls back
/// to host windows: the send and receive buffers of the verbs communicator
/// (2 x 3 windows x 48 MiB, kBufferBytes in
/// src/models/qwen38_flash_next/kernels/rocm/verbs.cpp) plus its flag page.
/// RLIMIT_MEMLOCK must cover it; 8 MiB, the default in ssh sessions and
/// systemd units, does not.
inline constexpr std::uint64_t kTp2RegisteredHostBytes =
    2ULL * 3ULL * (48ULL << 20) + 4096ULL;

/// What the TP2 rank preflight checks. The paths are parameters so the host
/// test can point them at a fake sysfs tree.
struct TpRdmaPreflightOptions {
  /// --tp-rdma-device; empty selects the host's only RDMA device, as the
  /// verbs communicator does.
  std::string device;
  std::uint32_t port{1};
  /// --tp-gid-index; unset selects as SelectRoceV2Gid does.
  std::optional<std::uint32_t> gid_index;
  std::uint64_t memlock_bytes{kTp2RegisteredHostBytes};
  std::filesystem::path sysfs_root{"/sys/class/infiniband"};
  std::filesystem::path verbs_class_root{"/sys/class/infiniband_verbs"};
  std::filesystem::path dev_root{"/dev/infiniband"};
  /// The soft RLIMIT_MEMLOCK in bytes (UINT64_MAX = unlimited); unset reads
  /// this process's limit.
  std::optional<std::uint64_t> memlock_limit;
};

struct TpRdmaPreflightResult {
  /// True when every check passed; the rank may start.
  bool ok{false};
  std::string device;
  std::string link_layer;
  std::optional<std::uint32_t> gid_index;
  std::string uverbs;
  /// One line per failed check: what is wrong and how to fix it.
  std::vector<std::string> failures;
  /// "device=... link=... gid_index=... uverbs=...".
  [[nodiscard]] std::string Summary() const;
};

/// The TP2 rank preflight: the RDMA device
/// exists, its port is ACTIVE, the GID the communicator will use is a set
/// RoCE v2 entry (RoCE) or entry 0 is set (InfiniBand), RLIMIT_MEMLOCK covers
/// the registered windows, and the device's uverbs node is there and opens
/// read-write for this user. Reads sysfs and /dev only: no verbs call, no
/// GPU, so it runs before the control channel and the model load and a rank
/// that cannot pair refuses at once with every reason instead of failing at
/// the handshake minutes later.
[[nodiscard]] TpRdmaPreflightResult RunTpRdmaPreflight(
    const TpRdmaPreflightOptions& options);

}  // namespace gufo::server

#endif  // GUFO_CLI_SERVE_TP_PREFLIGHT_HPP_
