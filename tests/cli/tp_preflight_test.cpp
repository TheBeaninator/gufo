#include "src/cli/serve/tp_preflight.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

namespace {

namespace fs = std::filesystem;
using gufo::server::RunTpRdmaPreflight;
using gufo::server::TpRdmaPreflightOptions;
using gufo::server::TpRdmaPreflightResult;

constexpr std::uint64_t kUnlimited = std::numeric_limits<std::uint64_t>::max();

int failures = 0;

void Check(bool condition, const std::string& what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

void Put(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream(path) << text << "\n";
}

bool Mentions(const TpRdmaPreflightResult& result, const std::string& text) {
  for (const auto& line : result.failures) {
    if (line.find(text) != std::string::npos) {
      return true;
    }
  }
  return false;
}

std::string Dump(const TpRdmaPreflightResult& result) {
  std::string text = result.Summary();
  for (const auto& line : result.failures) {
    text += "\n  " + line;
  }
  return text;
}

/// A fake host: /sys/class/infiniband, /sys/class/infiniband_verbs and
/// /dev/infiniband under one temporary root.
struct Host {
  fs::path root;
  Host() {
    char pattern[] = "/tmp/tp_preflight_test.XXXXXX";
    const char* made = ::mkdtemp(pattern);
    if (made == nullptr) {
      std::perror("mkdtemp");
      std::exit(2);
    }
    root = made;
    fs::create_directories(root / "sys" / "infiniband");
    fs::create_directories(root / "sys" / "infiniband_verbs");
    fs::create_directories(root / "dev");
  }
  ~Host() {
    std::error_code ec;
    fs::remove_all(root, ec);
  }
  [[nodiscard]] fs::path Port(const std::string& device, int port = 1) const {
    return root / "sys" / "infiniband" / device / "ports" / std::to_string(port);
  }
  /// The GID table a ConnectX RoCE port shows: link-local v1/v2 at 0
  /// and 1, the IPv4 address's v1/v2 at 2 and 3, the rest unset.
  void AddRoceDevice(const std::string& device, const std::string& uverbs,
                     const std::string& state = "4: ACTIVE") const {
    const fs::path port = Port(device);
    Put(port / "state", state);
    Put(port / "phys_state", state == "4: ACTIVE" ? "5: LinkUp" : "3: Disabled");
    Put(port / "link_layer", "Ethernet");
    Put(port / "gids" / "0", "fe80:0000:0000:0000:0200:5eff:fe00:5301");
    Put(port / "gids" / "1", "fe80:0000:0000:0000:0200:5eff:fe00:5301");
    Put(port / "gids" / "2", "0000:0000:0000:0000:0000:ffff:c000:021f");
    Put(port / "gids" / "3", "0000:0000:0000:0000:0000:ffff:c000:021f");
    Put(port / "gids" / "4", "0000:0000:0000:0000:0000:0000:0000:0000");
    Put(port / "gid_attrs" / "types" / "0", "IB/RoCE v1");
    Put(port / "gid_attrs" / "types" / "1", "RoCE v2");
    Put(port / "gid_attrs" / "types" / "2", "IB/RoCE v1");
    Put(port / "gid_attrs" / "types" / "3", "RoCE v2");
    if (!uverbs.empty()) {
      fs::create_directories(root / "sys" / "infiniband" / device / "device" /
                             "infiniband_verbs" / uverbs);
      AddNode(uverbs, 0666);
    }
  }
  void AddNode(const std::string& name, mode_t mode) const {
    Put(root / "dev" / name, "");
    ::chmod((root / "dev" / name).c_str(), mode);
  }
  [[nodiscard]] TpRdmaPreflightOptions Options() const {
    TpRdmaPreflightOptions options;
    options.sysfs_root = root / "sys" / "infiniband";
    options.verbs_class_root = root / "sys" / "infiniband_verbs";
    options.dev_root = root / "dev";
    options.memlock_limit = kUnlimited;
    return options;
  }
};

void HealthyRocePortPasses() {
  Host host;
  host.AddRoceDevice("rocep196s0", "uverbs0");
  const auto result = RunTpRdmaPreflight(host.Options());
  Check(result.ok, "healthy RoCE port passes: " + Dump(result));
  Check(result.device == "rocep196s0", "the only device is chosen");
  Check(result.gid_index == 3u, "auto GID is the IPv4 RoCE v2 entry 3");
  Check(result.link_layer == "Ethernet", "link layer read");
  Check(result.uverbs == (host.root / "dev" / "uverbs0").string(),
        "uverbs node found through device/infiniband_verbs");
}

void GidIndexChecks() {
  Host host;
  host.AddRoceDevice("mlx5_0", "uverbs0");
  auto options = host.Options();
  options.gid_index = 3;
  Check(RunTpRdmaPreflight(options).ok, "explicit GID 3 passes");
  options.gid_index = 2;
  auto result = RunTpRdmaPreflight(options);
  Check(!result.ok && Mentions(result, "not RoCE v2"),
        "GID 2 (RoCE v1) refused: " + Dump(result));
  options.gid_index = 7;
  result = RunTpRdmaPreflight(options);
  Check(!result.ok && Mentions(result, "GID index 7") &&
            Mentions(result, "unset"),
        "unset GID 7 refused: " + Dump(result));
  // Two IPv4 RoCE v2 entries and no index: ambiguous, as SelectRoceV2Gid.
  Put(host.Port("mlx5_0") / "gids" / "5",
      "0000:0000:0000:0000:0000:ffff:c633:641f");
  Put(host.Port("mlx5_0") / "gid_attrs" / "types" / "5", "RoCE v2");
  options.gid_index.reset();
  result = RunTpRdmaPreflight(options);
  Check(!result.ok && Mentions(result, "several RoCE v2 IPv4 GIDs (indices 3, 5)"),
        "ambiguous GIDs refused: " + Dump(result));
}

void MemlockRefused() {
  Host host;
  host.AddRoceDevice("mlx5_0", "uverbs0");
  auto options = host.Options();
  options.memlock_limit = 8ULL << 20;
  const auto result = RunTpRdmaPreflight(options);
  Check(!result.ok && result.failures.size() == 1 &&
            Mentions(result, "RLIMIT_MEMLOCK is 8 MiB") &&
            Mentions(result, "LimitMEMLOCK=infinity") &&
            Mentions(result, "ControlMaster"),
        "8 MiB memlock refused with the fix: " + Dump(result));
  options.memlock_limit = gufo::server::kTp2RegisteredHostBytes;
  Check(RunTpRdmaPreflight(options).ok, "memlock equal to the need passes");
}

void PortDownRefusedWithEveryReason() {
  Host host;
  host.AddRoceDevice("mlx5_0", "uverbs0", "1: DOWN");
  auto options = host.Options();
  options.memlock_limit = 64ULL << 10;
  const auto result = RunTpRdmaPreflight(options);
  Check(!result.ok && result.failures.size() == 2 &&
            Mentions(result, "mlx5_0 port 1 is 1: DOWN") &&
            Mentions(result, "not ACTIVE") && Mentions(result, "RLIMIT_MEMLOCK"),
        "port down and memlock both reported: " + Dump(result));
  options.memlock_limit = kUnlimited;
  options.port = 2;
  Check(Mentions(RunTpRdmaPreflight(options), "mlx5_0 port 2 does not exist"),
        "a missing port is named");
}

void DeviceSelection() {
  {
    Host host;
    const auto result = RunTpRdmaPreflight(host.Options());
    Check(!result.ok && Mentions(result, "no RDMA device"),
          "empty sysfs refused: " + Dump(result));
  }
  Host host;
  host.AddRoceDevice("mlx5_0", "uverbs0");
  host.AddRoceDevice("mlx5_1", "uverbs1");
  auto options = host.Options();
  auto result = RunTpRdmaPreflight(options);
  Check(!result.ok && Mentions(result, "several RDMA devices (mlx5_0, mlx5_1)") &&
            Mentions(result, "--tp-rdma-device"),
        "two devices and no name refused: " + Dump(result));
  options.device = "mlx5_1";
  result = RunTpRdmaPreflight(options);
  Check(result.ok && result.uverbs == (host.root / "dev" / "uverbs1").string(),
        "named device passes with its own uverbs: " + Dump(result));
  options.device = "mlx5_9";
  result = RunTpRdmaPreflight(options);
  Check(!result.ok && Mentions(result, "RDMA device mlx5_9 not found (have mlx5_0, mlx5_1)"),
        "unknown device refused: " + Dump(result));
}

void UverbsChecks() {
  {
    Host host;
    host.AddRoceDevice("mlx5_0", "");
    const auto result = RunTpRdmaPreflight(host.Options());
    Check(!result.ok && Mentions(result, "ib_uverbs"),
          "no uverbs node refused: " + Dump(result));
  }
  {
    // A software device lists its ibdev under the infiniband_verbs class.
    Host host;
    host.AddRoceDevice("rxe0", "");
    Put(host.root / "sys" / "infiniband_verbs" / "uverbs4" / "ibdev", "rxe0");
    host.AddNode("uverbs4", 0666);
    const auto result = RunTpRdmaPreflight(host.Options());
    Check(result.ok && result.uverbs == (host.root / "dev" / "uverbs4").string(),
          "class-listed uverbs found: " + Dump(result));
  }
  {
    Host host;
    host.AddRoceDevice("mlx5_0", "uverbs0");
    std::filesystem::remove(host.root / "dev" / "uverbs0");
    const auto result = RunTpRdmaPreflight(host.Options());
    Check(!result.ok && Mentions(result, "is missing"),
          "missing /dev node refused: " + Dump(result));
  }
  if (::geteuid() != 0) {
    Host host;
    host.AddRoceDevice("mlx5_0", "uverbs0");
    ::chmod((host.root / "dev" / "uverbs0").c_str(), 0000);
    const auto result = RunTpRdmaPreflight(host.Options());
    Check(!result.ok && Mentions(result, "read-write"),
          "unopenable /dev node refused: " + Dump(result));
  }
}

void InfiniBandUsesGidZero() {
  Host host;
  const fs::path port = host.Port("mlx4_0");
  Put(port / "state", "4: ACTIVE");
  Put(port / "link_layer", "InfiniBand");
  Put(port / "gids" / "0", "fe80:0000:0000:0000:0002:c903:0000:1234");
  fs::create_directories(host.root / "sys" / "infiniband" / "mlx4_0" / "device" /
                         "infiniband_verbs" / "uverbs0");
  host.AddNode("uverbs0", 0666);
  const auto result = RunTpRdmaPreflight(host.Options());
  Check(result.ok && result.gid_index == 0u,
        "InfiniBand port passes on GID 0: " + Dump(result));
}

}  // namespace

int main() {
  HealthyRocePortPasses();
  GidIndexChecks();
  MemlockRefused();
  PortDownRefusedWithEveryReason();
  DeviceSelection();
  UverbsChecks();
  InfiniBandUsesGidZero();
  if (failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("tp_preflight_test: all checks passed\n");
  return 0;
}
