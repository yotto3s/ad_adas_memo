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

ObligationStatus parseObligationStatus(const std::string& statusStr) {
  if (statusStr == "Valid") {
    return ObligationStatus::Valid;
  }
  if (statusStr == "Timeout") {
    return ObligationStatus::Timeout;
  }
  if (statusStr == "Unknown") {
    return ObligationStatus::Unknown;
  }
  return ObligationStatus::Failure;
}

std::optional<std::chrono::milliseconds>
parseDurationMs(const std::string& detail) {
  static const std::regex DURATION_REGEX(R"(([\d.]+)s)");
  std::smatch durMatch;
  if (!std::regex_search(detail, durMatch, DURATION_REGEX)) {
    return std::nullopt;
  }
  auto durStr = durMatch[1].str();
  double seconds = 0.0;
  auto [ptr, ec] =
      std::from_chars(durStr.data(), durStr.data() + durStr.size(), seconds);
  (void)ptr;
  if (ec != std::errc{}) {
    return std::nullopt;
  }
  return std::chrono::milliseconds(static_cast<int>(seconds * MS_PER_SECOND));
}

/// Find the why3 binary or return a single-error result.
llvm::Expected<std::string> findWhy3Binary(const std::string& why3Binary) {
  auto why3 = llvm::sys::findProgramByName(why3Binary);
  if (!why3) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "why3 binary not found");
  }
  return why3.get();
}

llvm::SmallVector<llvm::StringRef, 8> buildWhy3Args(const std::string& why3Path,
                                                    const std::string& mlwPath,
                                                    int timeoutSeconds,
                                                    std::string& timelimitArg) {
  timelimitArg = "--timelimit=" + std::to_string(timeoutSeconds);
  llvm::SmallVector<llvm::StringRef, 8> args;
  args.push_back(why3Path);
  args.push_back("prove");
  args.push_back("-P");
  args.push_back("z3");
  args.push_back(timelimitArg);
  args.push_back(mlwPath);
  return args;
}

/// Create a temp file for capturing Why3 stdout/stderr, or return an error.
llvm::Expected<llvm::SmallString<128>> createWhy3TempFile() {
  llvm::SmallString<128> outputPath; // NOLINT(readability-magic-numbers)
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile("why3out", "txt", outputPath);
  if (ec) {
    return llvm::createStringError(ec, "failed to create temp file");
  }
  return outputPath;
}

void removeFileWithWarning(const llvm::SmallString<128>& path) {
  auto removeEc = llvm::sys::fs::remove(path);
  if (removeEc) {
    llvm::errs() << "warning: could not remove temp file: " << path << "\n";
  }
}

std::string readAndRemoveTempFile(const llvm::SmallString<128>& outputPath) {
  auto bufOrErr = llvm::MemoryBuffer::getFile(outputPath);
  std::string output;
  if (bufOrErr) {
    output = (*bufOrErr)->getBuffer().str();
  }
  removeFileWithWarning(outputPath);
  return output;
}

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
  static const std::regex GOAL_REGEX(
      R"(Goal\s+(\S+)\.\s+(Valid|Timeout|Unknown)(?:\s+\(([^)]*)\))?)");

  // Match lines like: "Theory SafeAdd" to track which module/function
  // goals belong to.
  static const std::regex THEORY_REGEX(R"(Theory\s+(\S+))");

  std::string currentFuncName;

  std::istringstream stream(output);
  std::string line;
  while (std::getline(stream, line)) {
    // Track current theory/module context for function name attribution
    std::smatch theoryMatch;
    if (std::regex_search(line, theoryMatch, THEORY_REGEX)) {
      auto it = moduleToFuncMap.find(theoryMatch[1].str());
      currentFuncName =
          it != moduleToFuncMap.end() ? it->second : theoryMatch[1].str();
    }

    std::smatch match;
    if (std::regex_search(line, match, GOAL_REGEX)) {
      ObligationResult result;
      result.name = match[1].str();
      result.functionName = currentFuncName;
      result.status = parseObligationStatus(match[2].str());

      // Parse duration if present (e.g., "0.01s, 0 steps")
      if (match.size() > DETAIL_GROUP_INDEX &&
          match[DETAIL_GROUP_INDEX].matched) {
        if (auto dur = parseDurationMs(match[DETAIL_GROUP_INDEX].str())) {
          result.duration = *dur;
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
  auto why3PathResult = findWhy3Binary(why3Binary);
  if (!why3PathResult) {
    llvm::consumeError(why3PathResult.takeError());
    ObligationResult err;
    err.name = "why3_not_found";
    err.status = ObligationStatus::Failure;
    return {err};
  }
  std::string why3Path = std::move(*why3PathResult);

  std::string timelimitArg;
  auto args = buildWhy3Args(why3Path, mlwPath, timeoutSeconds, timelimitArg);

  auto tempFileResult = createWhy3TempFile();
  if (!tempFileResult) {
    llvm::consumeError(tempFileResult.takeError());
    ObligationResult err;
    err.name = "execution_error";
    err.status = ObligationStatus::Failure;
    return {err};
  }
  llvm::SmallString<128> outputPath = std::move(*tempFileResult);

  std::array<std::optional<llvm::StringRef>, 3> redirects = {
      // NOLINT(readability-magic-numbers)
      std::nullopt,                // stdin
      llvm::StringRef(outputPath), // stdout
      llvm::StringRef(outputPath), // stderr -> same file
  };

  int exitCode = llvm::sys::ExecuteAndWait(why3Path, args,
                                           /*Env=*/std::nullopt, redirects);

  std::string output = readAndRemoveTempFile(outputPath);

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
