#include "backend/Why3Runner.h"

#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdio>
#include <memory>
#include <regex>
#include <sstream>

namespace arcanum {

std::vector<ObligationResult> parseWhy3Output(const std::string& output) {
  std::vector<ObligationResult> results;

  // Match lines like: "    Goal <name>. Valid (0.01s, 0 steps)."
  // or: "    Goal <name>. Timeout."
  // or: "    Goal <name>. Unknown ("reason")."
  std::regex goalRegex(
      R"(Goal\s+(\S+)\.\s+(Valid|Timeout|Unknown)(?:\s+\(([^)]*)\))?)");

  std::istringstream stream(output);
  std::string line;
  while (std::getline(stream, line)) {
    std::smatch match;
    if (std::regex_search(line, match, goalRegex)) {
      ObligationResult result;
      result.name = match[1].str();

      auto statusStr = match[2].str();
      if (statusStr == "Valid") {
        result.status = ObligationStatus::Valid;
      } else if (statusStr == "Timeout") {
        result.status = ObligationStatus::Timeout;
      } else if (statusStr == "Unknown") {
        result.status = ObligationStatus::Unknown;
      } else {
        result.status = ObligationStatus::Failure;
      }

      // Parse duration if present (e.g., "0.01s, 0 steps")
      if (match.size() > 3 && match[3].matched) {
        std::regex durationRegex(R"(([\d.]+)s)");
        std::smatch durMatch;
        auto detailStr = match[3].str();
        if (std::regex_search(detailStr, durMatch, durationRegex)) {
          double seconds = std::stod(durMatch[1].str());
          result.duration = std::chrono::milliseconds(
              static_cast<int>(seconds * 1000));
        }
      }

      results.push_back(std::move(result));
    }
  }

  return results;
}

std::vector<ObligationResult> runWhy3(const std::string& mlwPath,
                                      const std::string& why3Binary,
                                      int timeoutSeconds) {
  // Find the why3 binary
  auto why3 = llvm::sys::findProgramByName(why3Binary);
  if (!why3) {
    ObligationResult err;
    err.name = "why3_not_found";
    err.status = ObligationStatus::Failure;
    return {err};
  }

  // Build command: why3 prove -P z3 --timelimit=<timeout> <file.mlw>
  std::string cmd = why3.get() + " prove -P z3 --timelimit=" +
                    std::to_string(timeoutSeconds) + " " + mlwPath +
                    " 2>&1";

  // Execute and capture output
  std::array<char, 4096> buffer;
  std::string output;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
                                                 pclose);
  if (!pipe) {
    ObligationResult err;
    err.name = "execution_error";
    err.status = ObligationStatus::Failure;
    return {err};
  }

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    output += buffer.data();
  }

  return parseWhy3Output(output);
}

} // namespace arcanum
