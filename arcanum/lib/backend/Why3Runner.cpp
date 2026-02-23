#include "arcanum/backend/Why3Runner.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <charconv>
#include <regex>
#include <sstream>

namespace arcanum {
namespace {
/// Regex capture group index for the duration/detail field in Why3 output.
constexpr size_t DETAIL_GROUP_INDEX = 3;
/// Conversion factor from seconds to milliseconds.
constexpr int MS_PER_SECOND = 1000;
} // namespace

/// Parse Why3 stdout into obligation results.
///
/// Format dependency: This parser relies on Why3's default textual output
/// format.  Expected patterns:
///   - "Theory <ModuleName>" lines to track function context
///   - "Goal <name>. Valid (Xs, N steps)." for proven goals
///   - "Goal <name>. Timeout." for timed-out goals
///   - "Goal <name>. Unknown (\"reason\")." for unknown goals
/// If Why3 changes its output format, this parser will silently produce
/// empty results.  Consider using Why3's JSON output (--json) in future
/// slices for robustness.
std::vector<ObligationResult>
parseWhy3Output(const std::string& output,
                const std::map<std::string, std::string>& moduleToFuncMap) {
  std::vector<ObligationResult> results;

  // Match lines like: "    Goal <name>. Valid (0.01s, 0 steps)."
  // or: "    Goal <name>. Timeout."
  // or: "    Goal <name>. Unknown ("reason")."
  static const std::regex goalRegex(
      R"(Goal\s+(\S+)\.\s+(Valid|Timeout|Unknown)(?:\s+\(([^)]*)\))?)");

  // Match lines like: "Theory SafeAdd" to track which module/function
  // goals belong to.
  static const std::regex theoryRegex(R"(Theory\s+(\S+))");

  std::string currentFuncName;

  std::istringstream stream(output);
  std::string line;
  while (std::getline(stream, line)) {
    // Track current theory/module context for function name attribution
    std::smatch theoryMatch;
    if (std::regex_search(line, theoryMatch, theoryRegex)) {
      auto it = moduleToFuncMap.find(theoryMatch[1].str());
      currentFuncName =
          it != moduleToFuncMap.end() ? it->second : theoryMatch[1].str();
    }

    std::smatch match;
    if (std::regex_search(line, match, goalRegex)) {
      ObligationResult result;
      result.name = match[1].str();
      result.functionName = currentFuncName;

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
      if (match.size() > DETAIL_GROUP_INDEX &&
          match[DETAIL_GROUP_INDEX].matched) {
        static const std::regex durationRegex(R"(([\d.]+)s)");
        std::smatch durMatch;
        auto detailStr = match[DETAIL_GROUP_INDEX].str();
        if (std::regex_search(detailStr, durMatch, durationRegex)) {
          auto durStr = durMatch[1].str();
          double seconds = 0.0;
          auto [ptr, ec] = std::from_chars(
              durStr.data(), durStr.data() + durStr.size(), seconds);
          (void)ptr;
          if (ec == std::errc{}) {
            result.duration = std::chrono::milliseconds(
                static_cast<int>(seconds * MS_PER_SECOND));
          }
        }
      }

      results.push_back(std::move(result));
    }
  }

  return results;
}

std::vector<ObligationResult>
runWhy3(const std::string& mlwPath,
        const std::map<std::string, std::string>& moduleToFuncMap,
        const std::string& why3Binary, int timeoutSeconds) {
  // Find the why3 binary
  auto why3 = llvm::sys::findProgramByName(why3Binary);
  if (!why3) {
    ObligationResult err;
    err.name = "why3_not_found";
    err.status = ObligationStatus::Failure;
    return {err};
  }

  // Build argument list for why3 prove
  llvm::SmallVector<llvm::StringRef, 8> args;
  args.push_back(why3.get());
  args.push_back("prove");
  args.push_back("-P");
  args.push_back("z3");
  std::string timelimitArg = "--timelimit=" + std::to_string(timeoutSeconds);
  args.push_back(timelimitArg);
  args.push_back(mlwPath);

  // Create temp file to capture stdout+stderr
  llvm::SmallString<128> outputPath; // NOLINT(readability-magic-numbers)
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile("why3out", "txt", outputPath);
  if (ec) {
    ObligationResult err;
    err.name = "execution_error";
    err.status = ObligationStatus::Failure;
    return {err};
  }

  std::array<std::optional<llvm::StringRef>, 3> redirects = {
      // NOLINT(readability-magic-numbers)
      std::nullopt,                // stdin
      llvm::StringRef(outputPath), // stdout
      llvm::StringRef(outputPath), // stderr -> same file
  };

  int exitCode = llvm::sys::ExecuteAndWait(why3.get(), args,
                                           /*Env=*/std::nullopt, redirects);

  // Read output file
  auto bufOrErr = llvm::MemoryBuffer::getFile(outputPath);
  std::string output;
  if (bufOrErr) {
    output = (*bufOrErr)->getBuffer().str();
  }
  auto removeEc = llvm::sys::fs::remove(outputPath);
  if (removeEc) {
    llvm::errs() << "warning: could not remove temp file: " << outputPath
                 << "\n";
  }

  // Check exit code: non-zero indicates Why3 crashed or had config errors.
  // Parse whatever output was produced but return Failure if no goals
  // were found and the exit code was non-zero.
  auto results = parseWhy3Output(output, moduleToFuncMap);
  if (exitCode != 0 && results.empty()) {
    ObligationResult err;
    err.name = "why3_error";
    err.status = ObligationStatus::Failure;
    return {err};
  }

  return results;
}

} // namespace arcanum
