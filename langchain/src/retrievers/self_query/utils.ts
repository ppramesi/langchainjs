import { isObject } from "../../util/type_utils.js";

/**
 * Checks if a provided filter is empty. The filter can be a function, an
 * object, a string, or undefined.
 */
export function isFilterEmpty(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  filter: ((q: any) => any) | object | string | undefined
): filter is undefined {
  if (!filter) return true;
  // for Milvus
  if (typeof filter === "string" && filter.length > 0) {
    return false;
  }
  if (typeof filter === "function") {
    return false;
  }
  return isObject(filter) && Object.keys(filter).length === 0;
}
