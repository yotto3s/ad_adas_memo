#include "arcanum/frontend/ContractParser.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Comment.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RawCommentList.h"
#include "clang/Basic/SourceManager.h"

#include "llvm/ADT/StringRef.h"

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
    if (matchString("<=")) {
      auto rhs = parseAddSub();
      if (!rhs) {
        return nullptr;
      }
      return ContractExpr::makeBinaryOp(BinaryOpKind::Le, lhs, rhs);
    }
    if (matchString(">=")) {
      auto rhs = parseAddSub();
      if (!rhs) {
        return nullptr;
      }
      return ContractExpr::makeBinaryOp(BinaryOpKind::Ge, lhs, rhs);
    }
    if (matchString("==")) {
      auto rhs = parseAddSub();
      if (!rhs) {
        return nullptr;
      }
      return ContractExpr::makeBinaryOp(BinaryOpKind::Eq, lhs, rhs);
    }
    if (matchString("!=")) {
      auto rhs = parseAddSub();
      if (!rhs) {
        return nullptr;
      }
      return ContractExpr::makeBinaryOp(BinaryOpKind::Ne, lhs, rhs);
    }
    if (matchString("<")) {
      auto rhs = parseAddSub();
      if (!rhs) {
        return nullptr;
      }
      return ContractExpr::makeBinaryOp(BinaryOpKind::Lt, lhs, rhs);
    }
    if (matchString(">")) {
      auto rhs = parseAddSub();
      if (!rhs) {
        return nullptr;
      }
      return ContractExpr::makeBinaryOp(BinaryOpKind::Gt, lhs, rhs);
    }
    return lhs;
  }

  ContractExprPtr parseAddSub() {
    auto lhs = parseMulDiv();
    if (!lhs) {
      return nullptr;
    }
    skipWhitespace();
    while (true) {
      if (matchString("+")) {
        auto rhs = parseMulDiv();
        if (!rhs) {
          return nullptr;
        }
        lhs = ContractExpr::makeBinaryOp(BinaryOpKind::Add, lhs, rhs);
      } else if (matchString("-")) {
        auto rhs = parseMulDiv();
        if (!rhs) {
          return nullptr;
        }
        lhs = ContractExpr::makeBinaryOp(BinaryOpKind::Sub, lhs, rhs);
      } else {
        break;
      }
    }
    return lhs;
  }

  ContractExprPtr parseMulDiv() {
    auto lhs = parseUnary();
    if (!lhs) {
      return nullptr;
    }
    skipWhitespace();
    while (true) {
      if (matchString("*")) {
        auto rhs = parseUnary();
        if (!rhs) {
          return nullptr;
        }
        lhs = ContractExpr::makeBinaryOp(BinaryOpKind::Mul, lhs, rhs);
      } else if (matchString("/")) {
        auto rhs = parseUnary();
        if (!rhs) {
          return nullptr;
        }
        lhs = ContractExpr::makeBinaryOp(BinaryOpKind::Div, lhs, rhs);
      } else if (matchString("%")) {
        auto rhs = parseUnary();
        if (!rhs) {
          return nullptr;
        }
        lhs = ContractExpr::makeBinaryOp(BinaryOpKind::Rem, lhs, rhs);
      } else {
        break;
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
    // Parenthesized expression
    if (text[pos] == '(') {
      ++pos;
      auto expr = parseOr();
      skipWhitespace();
      if (pos < text.size() && text[pos] == ')') {
        ++pos;
      }
      return expr;
    }
    // \result
    if (matchString("\\result")) {
      return ContractExpr::makeResultRef();
    }
    // true/false
    if (matchString("true")) {
      return ContractExpr::makeBoolLiteral(true);
    }
    if (matchString("false")) {
      return ContractExpr::makeBoolLiteral(false);
    }
    // Integer literal
    if (std::isdigit(static_cast<unsigned char>(text[pos])) != 0) {
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
    // Identifier (parameter name)
    if ((std::isalpha(static_cast<unsigned char>(text[pos])) != 0) ||
        text[pos] == '_') {
      size_t start = pos;
      while (pos < text.size() &&
             ((std::isalnum(static_cast<unsigned char>(text[pos])) != 0) ||
              text[pos] == '_')) {
        ++pos;
      }
      return ContractExpr::makeParamRef(text.substr(start, pos - start).str());
    }
    // Fallback: unrecognized token
    return nullptr;
  }

  llvm::StringRef text;
  size_t pos{0};
};

/// Extract //@ lines from a raw comment block and return them.
std::vector<std::string> extractAnnotationLines(llvm::StringRef commentText) {
  std::vector<std::string> lines;
  llvm::SmallVector<llvm::StringRef, 8>
      splitLines; // NOLINT(readability-magic-numbers)
  commentText.split(splitLines, '\n');

  for (auto& line : splitLines) {
    auto trimmed = line.trim();
    // Remove leading // if present
    if (trimmed.starts_with("//")) {
      trimmed = trimmed.drop_front(2).ltrim();
    }
    // Check for @ prefix
    if (trimmed.starts_with("@")) {
      trimmed = trimmed.drop_front(1).ltrim();
      lines.push_back(trimmed.str());
    }
  }
  return lines;
}

} // namespace

std::map<const clang::FunctionDecl*, ContractInfo>
parseContracts(clang::ASTContext& context) {
  std::map<const clang::FunctionDecl*, ContractInfo> result;
  auto& sm = context.getSourceManager();

  // Iterate over all function declarations
  for (auto* decl : context.getTranslationUnitDecl()->decls()) {
    auto* funcDecl = llvm::dyn_cast<clang::FunctionDecl>(decl);
    if ((funcDecl == nullptr) || !funcDecl->hasBody()) {
      continue;
    }

    // Get the raw comment associated with this declaration
    const auto* rawComment = context.getRawCommentForDeclNoCache(funcDecl);
    if (rawComment == nullptr) {
      continue;
    }

    auto commentText = rawComment->getRawText(sm);
    auto annotationLines = extractAnnotationLines(commentText);

    if (annotationLines.empty()) {
      continue;
    }

    ContractInfo info;
    for (const auto& line : annotationLines) {
      llvm::StringRef lineRef(line);
      if (lineRef.starts_with(REQUIRES_PREFIX)) {
        auto exprText = lineRef.drop_front(REQUIRES_PREFIX.size()).trim();
        ExprParser parser(exprText);
        if (auto expr = parser.parse()) {
          info.preconditions.push_back(std::move(expr));
        }
      } else if (lineRef.starts_with(ENSURES_PREFIX)) {
        auto exprText = lineRef.drop_front(ENSURES_PREFIX.size()).trim();
        ExprParser parser(exprText);
        if (auto expr = parser.parse()) {
          info.postconditions.push_back(std::move(expr));
        }
      }
    }

    if (!info.preconditions.empty() || !info.postconditions.empty()) {
      result[funcDecl] = std::move(info);
    }
  }

  return result;
}

} // namespace arcanum
