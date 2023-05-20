import {
  Comparator,
  Comparators,
  Comparison,
  Operation,
  Operators,
  StructuredQuery,
  VisitorStructuredQueryResult,
  FunctionFilter,
} from "../../chains/query_constructor/ir.js";
import { Document } from "../../document.js";
import { BaseTranslator } from "./translator.js";

type ValueType = {
  eq: string | number;
  neq: string | number;
  lt: string | number;
  lte: string | number;
  gt: string | number;
  gte: string | number;
  in: (string | number)[];
  nin: (string | number)[];
};

export class FunctionalTranslator extends BaseTranslator {
  formatFunction(): string {
    throw new Error("Not implemented");
  }

  getComparatorFunction<C extends Comparator>(
    comparator: Comparator
  ): (a: string | number, b: ValueType[C]) => boolean {
    switch (comparator) {
      case Comparators.eq: {
        return (a: string | number, b: ValueType[C]) => a === b;
      }
      case Comparators.neq: {
        return (a: string | number, b: ValueType[C]) => a !== b;
      }
      case Comparators.gt: {
        return (a: string | number, b: ValueType[C]) => a > b;
      }
      case Comparators.gte: {
        return (a: string | number, b: ValueType[C]) => a >= b;
      }
      case Comparators.lt: {
        return (a: string | number, b: ValueType[C]) => a < b;
      }
      case Comparators.lte: {
        return (a: string | number, b: ValueType[C]) => a <= b;
      }
      case Comparators.in: {
        return (a: string | number, b: ValueType[C]) => {
          if (!Array.isArray(b))
            throw new Error("Comparator value not an array");
          return b.includes(a);
        };
      }
      case Comparators.nin: {
        return (a: string | number, b: ValueType[C]) => {
          if (!Array.isArray(b))
            throw new Error("Comparator value not an array");
          return !b.includes(a);
        };
      }
      default: {
        throw new Error("Unknown comparator");
      }
    }
  }

  visitOperation(operation: Operation): FunctionFilter {
    const { operator, args } = operation;
    if (operator in this.allowedOperators) {
      if (operator === Operators.and) {
        return (document: Document) => {
          if (!args) {
            return true;
          }
          const result = args.reduce((acc, arg) => {
            const result = arg.accept(this);
            if (typeof result === "function") {
              return acc && result(document);
            } else {
              throw new Error("Filter is not a function");
            }
          }, true);
          return result;
        };
      } else if (operator === Operators.or) {
        return (document: Document) => {
          if (!args) {
            return true;
          }
          const result = args.reduce((acc, arg) => {
            const result = arg.accept(this);
            if (typeof result === "function") {
              return acc || result(document);
            } else {
              throw new Error("Filter is not a function");
            }
          }, false);
          return result;
        };
      } else {
        throw new Error("Unknown operator");
      }
    } else {
      throw new Error("Operator not allowed");
    }
  }

  visitComparison(comparison: Comparison): FunctionFilter {
    const { comparator, attribute, value } = comparison;
    const undefinedTrue = [Comparators.neq, Comparators.nin];
    if (this.allowedComparators.includes(comparator)) {
      const comparatorFunction = this.getComparatorFunction(comparator);
      return (document: Document) => {
        const documentValue = document.metadata[attribute];
        if (documentValue === undefined) {
          if (undefinedTrue.includes(comparator)) return true;
          return false;
        }
        return comparatorFunction(documentValue, value);
      };
    } else {
      throw new Error("Comparator not allowed");
    }
  }

  visitStructuredQuery(query: StructuredQuery): VisitorStructuredQueryResult {
    const filterFunction = query.filter?.accept(this);
    if (typeof filterFunction !== "function") {
      throw new Error("Structured query filter is not a function");
    }
    return { filter: filterFunction };
  }
}
