import type { ESTree } from "meriyah";
import { NodeHandler, ASTParser } from "./base.js";
import { CallExpressionType, MemberExpressionType } from "./types.js";
import { IdentifierHandler } from "./identifier_handler.js";

export class CallExpressionHandler extends NodeHandler {
  async accepts(node: ESTree.Node): Promise<ESTree.CallExpression | boolean> {
    if (ASTParser.isCallExpression(node)) {
      return node;
    } else {
      return false;
    }
  }

  async handle(node: ESTree.CallExpression): Promise<CallExpressionType> {
    function checkCallExpressionArgumentType(arg: ESTree.Node): boolean {
      return [
        ASTParser.isStringLiteral,
        ASTParser.isNumericLiteral,
        ASTParser.isBooleanLiteral,
        ASTParser.isArrayExpression,
        ASTParser.isObjectExpression,
        ASTParser.isCallExpression,
        ASTParser.isIdentifier,
      ].reduce((acc, func) => acc || func(arg), false);
    }
    if (this.parentHandler === undefined) {
      throw new Error(
        "ArrayLiteralExpressionHandler must have a parent handler"
      );
    }
    const { callee } = node;
    let funcCall;
    if (ASTParser.isIdentifier(callee)) {
      const identifierHandler = new IdentifierHandler(this.parentHandler);
      const identifier = await identifierHandler.handle(callee);
      funcCall = identifier.value;
    } else if (ASTParser.isMemberExpression(callee)) {
      funcCall = (await this.parentHandler.handle(
        callee as ESTree.MemberExpression
      )) as MemberExpressionType;
    } else {
      throw new Error("Unknown expression type");
    }

    const args = await Promise.all(
      node.arguments.map((arg) => {
        if (!checkCallExpressionArgumentType(arg)) {
          throw new Error("Unknown argument type");
        }
        if (!this.parentHandler) {
          throw new Error("CallExpressionHandler must have a parent handler");
        }
        return this.parentHandler.handle(arg as ESTree.Node);
      })
    );
    return { type: "call_expression", funcCall, args };
  }
}
