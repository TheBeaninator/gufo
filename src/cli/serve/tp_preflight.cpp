#include "src/cli/serve/tp_preflight.hpp"

#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>

namespace gufo::server {
namespace {

namespace fs = std::filesystem;

/// The first line of a sysfs attribute, or nullopt when it cannot be read
/// (an unset GID's type reads EINVAL).
std::optional<std::string> ReadLine(const fs::path& path) {
  std::ifstream in(path);
  if (!in) {
    return std::nullopt;
  }
  std::string line;
  if (!std::getline(in, line)) {
    return std::nullopt;
  }
  while (!line.empty() &&
         std::isspace(static_cast<unsigned char>(line.back())) != 0) {
    line.pop_back();
  }
  return line;
}

std::vector<std::string> Children(const fs::path& dir) {
  std::vector<std::string> names;
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    names.push_back(entry.path().filename().string());
  }
  std::sort(names.begin(), names.end());
  return names;
}

std::string Join(const std::vector<std::string>& items) {
  std::string text;
  for (const auto& item : items) {
    text += (text.empty() ? "" : ", ") + item;
  }
  return text;
}

bool AllDigits(const std::string& text) {
  return !text.empty() &&
         std::all_of(text.begin(), text.end(), [](unsigned char c) {
           return std::isdigit(c) != 0;
         });
}

/// A GID as sysfs prints it: eight groups of four hex digits.
bool GidUnset(const std::string& gid) {
  return std::all_of(gid.begin(), gid.end(),
                     [](char c) { return c == '0' || c == ':'; });
}

/// ::ffff:a.b.c.d, the RoCE v2 GID of an IPv4 address of the netdev.
bool GidIpv4Mapped(const std::string& gid) {
  return gid.rfind("0000:0000:0000:0000:0000:ffff:", 0) == 0;
}

std::string Bytes(std::uint64_t bytes) {
  if (bytes == std::numeric_limits<std::uint64_t>::max()) {
    return "unlimited";
  }
  if (bytes >= (1ULL << 20)) {
    // Rounded up: a limit just short of the need never reads as enough.
    return std::to_string((bytes + (1ULL << 20) - 1) >> 20) + " MiB";
  }
  return std::to_string(bytes) + " bytes";
}

std::uint64_t ProcessMemlockLimit() {
  rlimit limit{};
  if (::getrlimit(RLIMIT_MEMLOCK, &limit) != 0) {
    return 0;
  }
  return limit.rlim_cur == RLIM_INFINITY
             ? std::numeric_limits<std::uint64_t>::max()
             : static_cast<std::uint64_t>(limit.rlim_cur);
}

struct GidRow {
  std::uint32_t index{0};
  std::string gid;
  std::string type;  // "RoCE v2", "IB/RoCE v1", or empty when unreadable
};

std::vector<GidRow> ReadGids(const fs::path& port_dir) {
  std::vector<GidRow> rows;
  for (const auto& name : Children(port_dir / "gids")) {
    if (!AllDigits(name) || name.size() > 6) {
      continue;
    }
    const auto gid = ReadLine(port_dir / "gids" / name);
    if (!gid.has_value() || GidUnset(*gid)) {
      continue;
    }
    rows.push_back({.index = static_cast<std::uint32_t>(std::stoul(name)),
                    .gid = *gid,
                    .type = ReadLine(port_dir / "gid_attrs" / "types" / name)
                                .value_or("")});
  }
  std::sort(rows.begin(), rows.end(),
            [](const GidRow& a, const GidRow& b) { return a.index < b.index; });
  return rows;
}

}  // namespace

std::string TpRdmaPreflightResult::Summary() const {
  std::ostringstream out;
  out << "device=" << (device.empty() ? "?" : device)
      << " link=" << (link_layer.empty() ? "?" : link_layer) << " gid_index="
      << (gid_index.has_value() ? std::to_string(*gid_index) : "?")
      << " uverbs=" << (uverbs.empty() ? "?" : uverbs);
  return out.str();
}

TpRdmaPreflightResult RunTpRdmaPreflight(
    const TpRdmaPreflightOptions& options) {
  TpRdmaPreflightResult result;
  auto& failures = result.failures;

  // Memory lock first: it needs no device and is the most common miss.
  const std::uint64_t memlock =
      options.memlock_limit.value_or(ProcessMemlockLimit());
  if (memlock < options.memlock_bytes) {
    failures.push_back(
        "RLIMIT_MEMLOCK is " + Bytes(memlock) + "; TP2 registers " +
        Bytes(options.memlock_bytes) +
        " of host windows with the RDMA device: start the rank with "
        "LimitMEMLOCK=infinity (systemd-run -p LimitMEMLOCK=infinity, or "
        "memlock unlimited in limits.d and a fresh login; an ssh "
        "ControlMaster session that predates the change keeps the old "
        "limit)");
  }

  // The device: the named one, or the host's only one.
  const auto devices = Children(options.sysfs_root);
  if (devices.empty()) {
    failures.push_back("no RDMA device: " + options.sysfs_root.string() +
                       " is empty (load the NIC's RDMA driver, e.g. "
                       "mlx5_ib, or the USB4 verbs driver)");
    return result;
  }
  if (options.device.empty()) {
    if (devices.size() != 1) {
      failures.push_back("several RDMA devices (" + Join(devices) +
                         "): choose one with --tp-rdma-device");
      return result;
    }
    result.device = devices.front();
  } else if (std::find(devices.begin(), devices.end(), options.device) ==
             devices.end()) {
    failures.push_back("RDMA device " + options.device + " not found (have " +
                       Join(devices) + ")");
    return result;
  } else {
    result.device = options.device;
  }
  const fs::path device_dir = options.sysfs_root / result.device;
  const fs::path port_dir =
      device_dir / "ports" / std::to_string(options.port);
  const std::string where =
      result.device + " port " + std::to_string(options.port);

  // Port state.
  const auto state = ReadLine(port_dir / "state");
  if (!state.has_value()) {
    failures.push_back(where + " does not exist (--tp-rdma-port)");
    return result;
  }
  if (state->find("ACTIVE") == std::string::npos) {
    const auto phys = ReadLine(port_dir / "phys_state").value_or("?");
    failures.push_back(where + " is " + *state + " (physical " + phys +
                       "), not ACTIVE: check the cable, the switch port and "
                       "that the netdev is up");
  }
  result.link_layer = ReadLine(port_dir / "link_layer").value_or("");
  const bool roce = result.link_layer == "Ethernet";
  if (!roce && result.link_layer != "InfiniBand") {
    failures.push_back(where + " link layer is '" + result.link_layer +
                       "', neither Ethernet (RoCE) nor InfiniBand");
  }

  // GID: the entry the communicator will use (see SelectGid in verbs.cpp
  // and SelectRoceV2Gid in rdma_gid.cpp, whose rule this mirrors).
  const auto gids = ReadGids(port_dir);
  const auto find_gid = [&](std::uint32_t index) -> const GidRow* {
    for (const auto& row : gids) {
      if (row.index == index) {
        return &row;
      }
    }
    return nullptr;
  };
  if (options.gid_index.has_value() || !roce) {
    const std::uint32_t index = options.gid_index.value_or(0);
    const GidRow* row = find_gid(index);
    if (row == nullptr) {
      failures.push_back("GID index " + std::to_string(index) + " of " +
                         where +
                         " is unset: give the port's netdev an address, or "
                         "choose a set entry with --tp-gid-index");
    } else if (roce && row->type != "RoCE v2") {
      failures.push_back("GID index " + std::to_string(index) + " of " +
                         where + " is '" + row->type + "' (" + row->gid +
                         "), not RoCE v2: TP2 uses RoCE v2 only; the RoCE "
                         "v2 entry of the link's IPv4 address is usually "
                         "index 3");
    } else {
      result.gid_index = index;
    }
  } else {
    std::vector<std::uint32_t> any;
    std::vector<std::uint32_t> ipv4;
    for (const auto& row : gids) {
      if (row.type != "RoCE v2") {
        continue;
      }
      any.push_back(row.index);
      if (GidIpv4Mapped(row.gid)) {
        ipv4.push_back(row.index);
      }
    }
    const auto& candidates = ipv4.empty() ? any : ipv4;
    if (candidates.size() == 1) {
      result.gid_index = candidates.front();
    } else {
      std::vector<std::string> names;
      for (const auto index : candidates) {
        names.push_back(std::to_string(index));
      }
      failures.push_back(
          candidates.empty()
              ? where + " has no RoCE v2 GID: give its netdev an address"
              : where + " has several RoCE v2 " +
                    (ipv4.empty() ? "GIDs" : "IPv4 GIDs") + " (indices " +
                    Join(names) + "): choose one with --tp-gid-index");
    }
  }

  // uverbs: libibverbs opens /dev/infiniband/uverbsN of this device.
  std::string uverbs_name;
  for (const auto& name : Children(device_dir / "device" / "infiniband_verbs")) {
    if (name.rfind("uverbs", 0) == 0) {
      uverbs_name = name;
    }
  }
  if (uverbs_name.empty()) {
    // Devices without a parent bus device (software verbs) list their
    // ibdev under the infiniband_verbs class instead.
    for (const auto& name : Children(options.verbs_class_root)) {
      if (ReadLine(options.verbs_class_root / name / "ibdev").value_or("") ==
          result.device) {
        uverbs_name = name;
      }
    }
  }
  if (uverbs_name.empty()) {
    failures.push_back("no uverbs node for " + result.device +
                       ": load ib_uverbs (the kernel's user verbs module)");
  } else {
    const fs::path node = options.dev_root / uverbs_name;
    result.uverbs = node.string();
    std::error_code ec;
    if (!fs::exists(node, ec)) {
      failures.push_back(node.string() + " is missing (udev did not create "
                                         "it; is /dev/infiniband mounted "
                                         "into the container?)");
    } else if (::access(node.c_str(), R_OK | W_OK) != 0) {
      failures.push_back(node.string() +
                         " does not open read-write for this user: add the "
                         "user to the group that owns it or fix the udev "
                         "rule");
    }
  }

  result.ok = failures.empty();
  return result;
}

}  // namespace gufo::server
