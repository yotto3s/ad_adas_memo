#include "arcanum/frontend/ContractParser.h"

#include <array>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Comment.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RawCommentList.h"
#include "clang/Basic/SourceManager.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <optional>
#include <sstream>

namespace arcanum {
namespace {
/// Base for decimal integer parsing.
constexpr unsigned DECIMAL_BASE = 10;
/// Prefix strings for contract annotation lines.
constexpr llvm::StringLiteral REQUIRES_PREFIX("requires:");
constexpr llvm::StringLiteral ENSURES_PREFIX("ensures:");
constexpr llvm::StringLiteral OVERFLOW_PREFIX("overflow:");
constexpr llvm::StringLiteral LOOP_INVARIANT_PREFIX("loop_invariant:");
constexpr llvm::StringLiteral LOOP_VARIANT_PREFIX("loop_variant:");
constexpr llvm::StringLiteral LOOP_ASSIGNS_PREFIX("loop_assigns:");
constexpr llvm::StringLiteral LABEL_PREFIX("label:");
} // namespace

ContractExprPtr ContractExpr::makeIntLiteral(int64_t val) {
  auto e = std::make_shared<ContractExpr>();
  e->kind = ContractExprKind::IntLiteral;
  e->intValue = val;
  return e;
}

ContractExprPtr ContractExpr::makeBoolLiteral(bool val) {
  auto e = std::make_shared<ContractExpr>();
  e->kind = ContractExprKind::BoolLiteral;
  e->boolValue = val;
  return e;
}

ContractExprPtr ContractExpr::makeParamRef(const std::string& name) {
  auto e = std::make_shared<ContractExpr>();
  e->kind = ContractExprKind::ParamRef;
  e->paramName = name;
  return e;
}

ContractExprPtr ContractExpr::makeResultRef() {
  auto e = std::make_shared<ContractExpr>();
  e->kind = ContractExprKind::ResultRef;
  return e;
}

ContractExprPtr ContractExpr::makeBinaryOp(BinaryOpKind op, ContractExprPtr lhs,
                                           ContractExprPtr rhs) {
  auto e = std::make_shared<ContractExpr>();
  e->kind = ContractExprKind::BinaryOp;
  e->binaryOp = op;
  e->left = std::move(lhs);
  e->right = std::move(rhs);
  return e;
}

ContractExprPtr ContractExpr::makeUnaryOp(UnaryOpKind op,
                                          ContractExprPtr operand) {
  auto e = std::make_shared<ContractExpr>();
  e->kind = ContractExprKind::UnaryOp;
  e->unaryOp = op;
  e->operand = std::move(operand);
  return e;
}

namespace {

/// Simple recursive-descent parser for contract expressions.
class ExprParser {
public:
  explicit ExprParser(llvm::StringRef text) : text(text) {}

  ContractExprPtr parse() {
    auto expr = parseOr();
    skipWhitespace();
    return expr;
  }

private:
  void skipWhitespace() {
    while (pos < text.size() &&
           (std::isspace(static_cast<unsigned char>(text[pos])) != 0)) {
      ++pos;
    }
  }

  bool matchString(llvm::StringRef s) {
    skipWhitespace();
    if (text.substr(pos).starts_with(s)) {
      pos += s.size();
      return true;
    }
    return false;
  }

  // --- Predicates ---

  [[nodiscard]] bool atOpenParen() const {
    return pos < text.size() && text[pos] == '(';
  }

  [[nodiscard]] bool atDigit() const {
    return pos < text.size() &&
           (std::isdigit(static_cast<unsigned char>(text[pos])) != 0);
  }

  [[nodiscard]] bool atIdentStart() const {
    return pos < text.size() &&
           ((std::isalpha(static_cast<unsigned char>(text[pos])) != 0) ||
            text[pos] == '_');
  }

  // --- Primary sub-parsers ---

  ContractExprPtr parseParenExpr() {
    ++pos; // consume '('
    auto expr = parseOr();
    skipWhitespace();
    if (pos < text.size() && text[pos] == ')') {
      ++pos;
    } else {
      // Missing closing parenthesis -- parse error
      return nullptr;
    }
    return expr;
  }

  /// Parse a bool literal keyword ("true"/"false") with word-boundary check.
  /// Returns nullptr if the keyword is immediately followed by an alnum or '_'
  /// (in which case the caller should treat the text as an identifier).
  ContractExprPtr parseBoolLiteral() {
    static constexpr size_t TRUE_LEN = 4;
    static constexpr size_t FALSE_LEN = 5;

    if (matchString("true")) {
      if (pos < text.size() &&
          ((std::isalnum(static_cast<unsigned char>(text[pos])) != 0) ||
           text[pos] == '_')) {
        pos -= TRUE_LEN; // backtrack: not a standalone keyword
        return nullptr;
      }
      return ContractExpr::makeBoolLiteral(true);
    }
    if (matchString("false")) {
      if (pos < text.size() &&
          ((std::isalnum(static_cast<unsigned char>(text[pos])) != 0) ||
           text[pos] == '_')) {
        pos -= FALSE_LEN; // backtrack: not a standalone keyword
        return nullptr;
      }
      return ContractExpr::makeBoolLiteral(false);
    }
    return nullptr;
  }

  ContractExprPtr parseIntLiteral() {
    size_t start = pos;
    while (pos < text.size() &&
           (std::isdigit(static_cast<unsigned char>(text[pos])) != 0)) {
      ++pos;
    }
    int64_t val = 0;
    auto sub = text.substr(start, pos - start);
    if (sub.getAsInteger(DECIMAL_BASE, val)) {
      return nullptr; // Error: invalid integer literal
    }
    return ContractExpr::makeIntLiteral(val);
  }

  ContractExprPtr parseIdentifier() {
    size_t start = pos;
    while (pos < text.size() &&
           ((std::isalnum(static_cast<unsigned char>(text[pos])) != 0) ||
            text[pos] == '_')) {
      ++pos;
    }
    return ContractExpr::makeParamRef(text.substr(start, pos - start).str());
  }

  // --- Grammar rules ---

  ContractExprPtr parseOr() {
    auto lhs = parseAnd();
    if (!lhs) {
      return nullptr;
    }
    while (matchString("||")) {
      auto rhs = parseAnd();
      if (!rhs) {
        return nullptr;
      }
      lhs = ContractExpr::makeBinaryOp(BinaryOpKind::Or, lhs, rhs);
    }
    return lhs;
  }

  ContractExprPtr parseAnd() {
    auto lhs = parseComparison();
    if (!lhs) {
      return nullptr;
    }
    while (matchString("&&")) {
      auto rhs = parseComparison();
      if (!rhs) {
        return nullptr;
      }
      lhs = ContractExpr::makeBinaryOp(BinaryOpKind::And, lhs, rhs);
    }
    return lhs;
  }

  ContractExprPtr parseComparison() {
    auto lhs = parseAddSub();
    if (!lhs) {
      return nullptr;
    }
    skipWhitespace();
    // Data-driven comparison operator table.  Multi-character operators
    // must appear before their single-character prefixes (e.g. "<=" before
    // "<").
    struct CompOp {
      llvm::StringLiteral token;
      BinaryOpKind kind;
    };
    static constexpr std::array<CompOp, 6> COMP_OPS = {{
        {.token = "<=", .kind = BinaryOpKind::Le},
        {.token = ">=", .kind = BinaryOpKind::Ge},
        {.token = "==", .kind = BinaryOpKind::Eq},
        {.token = "!=", .kind = BinaryOpKind::Ne},
        {.token = "<", .kind = BinaryOpKind::Lt},
        {.token = ">", .kind = BinaryOpKind::Gt},
    }};
    for (const auto& op : COMP_OPS) {
      if (matchString(op.token)) {
        auto rhs = parseAddSub();
        if (!rhs) {
          return nullptr;
        }
        return ContractExpr::makeBinaryOp(op.kind, lhs, rhs);
      }
    }
    return lhs;
  }

  ContractExprPtr parseAddSub() {
    auto lhs = parseMulDiv();
    if (!lhs) {
      return nullptr;
    }
    // Data-driven additive operator table.
    struct AddOp {
      llvm::StringLiteral token;
      BinaryOpKind kind;
    };
    static constexpr std::array<AddOp, 2> ADD_OPS = {{
        {.token = "+", .kind = BinaryOpKind::Add},
        {.token = "-", .kind = BinaryOpKind::Sub},
    }};
    skipWhitespace();
    bool matched = true;
    while (matched) {
      matched = false;
      for (const auto& op : ADD_OPS) {
        if (matchString(op.token)) {
          auto rhs = parseMulDiv();
          if (!rhs) {
            return nullptr;
          }
          lhs = ContractExpr::makeBinaryOp(op.kind, lhs, rhs);
          matched = true;
          break;
        }
      }
    }
    return lhs;
  }

  ContractExprPtr parseMulDiv() {
    auto lhs = parseUnary();
    if (!lhs) {
      return nullptr;
    }
    // Data-driven multiplicative operator table.
    struct MulOp {
      llvm::StringLiteral token;
      BinaryOpKind kind;
    };
    static constexpr std::array<MulOp, 3> MUL_OPS = {{
        {.token = "*", .kind = BinaryOpKind::Mul},
        {.token = "/", .kind = BinaryOpKind::Div},
        {.token = "%", .kind = BinaryOpKind::Rem},
    }};
    skipWhitespace();
    bool matched = true;
    while (matched) {
      matched = false;
      for (const auto& op : MUL_OPS) {
        if (matchString(op.token)) {
          auto rhs = parseUnary();
          if (!rhs) {
            return nullptr;
          }
          lhs = ContractExpr::makeBinaryOp(op.kind, lhs, rhs);
          matched = true;
          break;
        }
      }
    }
    return lhs;
  }

  ContractExprPtr parseUnary() {
    skipWhitespace();
    if (matchString("!")) {
      auto operand = parseUnary();
      if (!operand) {
        return nullptr;
      }
      return ContractExpr::makeUnaryOp(UnaryOpKind::Not, operand);
    }
    if (matchString("-")) {
      auto operand = parsePrimary();
      if (!operand) {
        return nullptr;
      }
      return ContractExpr::makeUnaryOp(UnaryOpKind::Neg, operand);
    }
    return parsePrimary();
  }

  ContractExprPtr parsePrimary() {
    skipWhitespace();
    if (pos >= text.size()) {
      return nullptr; // Error: unexpected end of expression
    }
    if (atOpenParen()) {
      return parseParenExpr();
    }
    if (matchString("\\result")) {
      return ContractExpr::makeResultRef();
    }
    if (auto boolExpr = parseBoolLiteral()) {
      return boolExpr;
    }
    if (atDigit()) {
      return parseIntLiteral();
    }
    if (atIdentStart()) {
      return parseIdentifier();
    }
    // Fallback: unrecognized token
    return nullptr;
  }

  llvm::StringRef text;
  size_t pos{0};
};

/// Extract the annotation payload from a single comment line, if any.
/// Returns the text after the "//@" prefix, trimmed, or std::nullopt if the
/// line is not an annotation line.
std::optional<std::string> extractAnnotationPayload(llvm::StringRef line) {
  auto trimmed = line.trim();
  // Remove leading // if present
  if (trimmed.starts_with("//")) {
    trimmed = trimmed.drop_front(2).ltrim();
  }
  // Check for @ prefix
  if (!trimmed.starts_with("@")) {
    return std::nullopt;
  }
  return trimmed.drop_front(1).ltrim().str();
}

} // namespace

/// Extract //@ lines from a raw comment block and return them.
std::vector<std::string> extractAnnotationLines(llvm::StringRef commentText) {
  std::vector<std::string> lines;
  llvm::SmallVector<llvm::StringRef, 8> // NOLINT(readability-magic-numbers)
      splitLines;
  commentText.split(splitLines, '\n');

  for (auto& line : splitLines) {
    if (auto payload = extractAnnotationPayload(line)) {
      lines.push_back(std::move(*payload));
    }
  }
  return lines;
}

/// Parse comma-separated identifiers from a string.
static std::vector<std::string>
parseCommaSeparatedIdents(llvm::StringRef text) {
  std::vector<std::string> result;
  llvm::SmallVector<llvm::StringRef, 8> parts;
  text.split(parts, ',');
  for (auto& part : parts) {
    auto trimmed = part.trim();
    if (!trimmed.empty()) {
      result.push_back(trimmed.str());
    }
  }
  return result;
}

void applyLoopAnnotationLine(llvm::StringRef line, LoopContractInfo& info) {
  if (line.starts_with(LOOP_INVARIANT_PREFIX)) {
    auto expr = line.drop_front(LOOP_INVARIANT_PREFIX.size()).trim();
    if (!info.invariant.empty()) {
      info.invariant += " && ";
    }
    info.invariant += expr.str();
  } else if (line.starts_with(LOOP_VARIANT_PREFIX)) {
    info.variant = line.drop_front(LOOP_VARIANT_PREFIX.size()).trim().str();
  } else if (line.starts_with(LOOP_ASSIGNS_PREFIX)) {
    auto assigns = line.drop_front(LOOP_ASSIGNS_PREFIX.size()).trim();
    info.assigns = parseCommaSeparatedIdents(assigns);
  } else if (line.starts_with(LABEL_PREFIX)) {
    info.label = line.drop_front(LABEL_PREFIX.size()).trim().str();
  }
}

namespace {

/// Retrieve the raw annotation lines for a function declaration.
/// Returns an empty vector if the declaration has no annotation comment.
std::vector<std::string>
getAnnotationLinesForDecl(clang::ASTContext& context, clang::SourceManager& sm,
                          const clang::FunctionDecl* funcDecl) {
  const auto* rawComment = context.getRawCommentForDeclNoCache(funcDecl);
  if (rawComment == nullptr) {
    return {};
  }
  return extractAnnotationLines(rawComment->getRawText(sm));
}

/// Parse a requires/ensures clause expression and append it to the target
/// list, or emit a warning on failure.
void parseExprClause(llvm::StringRef exprText, llvm::StringRef funcName,
                     unsigned funcLine, llvm::StringRef clauseName,
                     std::vector<ContractExprPtr>& target) {
  ExprParser parser(exprText);
  if (auto expr = parser.parse()) {
    target.push_back(std::move(expr));
  } else {
    llvm::errs() << "warning: in function '" << funcName << "' (line "
                 << funcLine << "): failed to parse '" << clauseName
                 << "' clause: '" << exprText
                 << "' (contract will be ignored)\n";
  }
}

/// Dispatch a single annotation line into the appropriate ContractInfo field.
void applyAnnotationLine(llvm::StringRef lineRef, ContractInfo& info,
                         llvm::StringRef funcName, unsigned funcLine) {
  if (lineRef.starts_with(REQUIRES_PREFIX)) {
    auto exprText = lineRef.drop_front(REQUIRES_PREFIX.size()).trim();
    parseExprClause(exprText, funcName, funcLine, "requires",
                    info.preconditions);
  } else if (lineRef.starts_with(ENSURES_PREFIX)) {
    auto exprText = lineRef.drop_front(ENSURES_PREFIX.size()).trim();
    parseExprClause(exprText, funcName, funcLine, "ensures",
                    info.postconditions);
  } else if (lineRef.starts_with(OVERFLOW_PREFIX)) {
    auto modeText = lineRef.drop_front(OVERFLOW_PREFIX.size()).trim();
    if (modeText == "trap" || modeText == "wrap" || modeText == "saturate") {
      info.overflowMode = modeText.str();
    } else {
      // SC-6: Emit diagnostic error (not just stderr warning) for invalid
      // overflow mode, consistent with other contract parsing errors.
      llvm::errs() << "error: in function '" << funcName << "' (line "
                   << funcLine << "): unknown overflow mode '" << modeText
                   << "'; expected 'trap', 'wrap', or 'saturate' "
                      "(defaulting to 'trap')\n";
    }
  }
}

/// Returns true when ContractInfo contains anything beyond the defaults that
/// warrants inserting it into the result map.
bool hasNonDefaultContracts(const ContractInfo& info) {
  return !info.preconditions.empty() || !info.postconditions.empty() ||
         info.overflowMode != "trap";
}

} // namespace

std::map<const clang::FunctionDecl*, ContractInfo>
parseContracts(clang::ASTContext& context) {
  std::map<const clang::FunctionDecl*, ContractInfo> result;
  auto& sm = context.getSourceManager();

  // Iterate over top-level TU declarations only.
  // IMPORTANT: This must iterate the same declaration list as
  // ArcLowering::lower() in Lowering.cpp, because the contract map is
  // keyed by FunctionDecl pointer identity.  Both functions must agree
  // on which declarations they process.  SubsetEnforcer also validates
  // that only TU-level functions are accepted (rejects namespaced functions).
  for (auto* decl : context.getTranslationUnitDecl()->decls()) {
    auto* funcDecl = llvm::dyn_cast<clang::FunctionDecl>(decl);
    if ((funcDecl == nullptr) || !funcDecl->hasBody()) {
      continue;
    }

    auto annotationLines = getAnnotationLinesForDecl(context, sm, funcDecl);
    if (annotationLines.empty()) {
      continue;
    }

    auto funcName = funcDecl->getNameAsString();
    auto funcLoc = funcDecl->getLocation();
    unsigned funcLine = 0;
    if (funcLoc.isValid()) {
      funcLine = sm.getPresumedLineNumber(funcLoc);
    }

    ContractInfo info;
    for (const auto& line : annotationLines) {
      applyAnnotationLine(llvm::StringRef(line), info, funcName, funcLine);
    }

    if (hasNonDefaultContracts(info)) {
      result[funcDecl] = std::move(info);
    }
  }

  return result;
}

} // namespace arcanum
