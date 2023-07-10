import * as uuid from "uuid";
import {
  AgentAction,
  AgentFinish,
  BaseMessage,
  ChainValues,
  LLMResult,
} from "../schema/index.js";
import {
  Serializable,
  Serialized,
  SerializedNotImplemented,
} from "../load/serializable.js";
import { SerializedFields } from "../load/map_keys.js";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Error = any;

export interface BaseCallbackHandlerInput {
  ignoreLLM?: boolean;
  ignoreChain?: boolean;
  ignoreAgent?: boolean;
  ignoreEmbeddings?: boolean;
}

export interface NewTokenIndices {
  prompt: number;
  completion: number;
}

abstract class BaseCallbackHandlerMethodsClass {
  /**
   * Called at the start of an LLM or Chat Model run, with the prompt(s)
   * and the run ID.
   */
  handleLLMStart?(
    llm: Serialized,
    prompts: string[],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    tags?: string[],
    metadata?: Record<string, unknown>
  ): Promise<void> | void;

  /**
   * Called when an LLM/ChatModel in `streaming` mode produces a new token
   */
  handleLLMNewToken?(
    token: string,
    /**
     * idx.prompt is the index of the prompt that produced the token
     *   (if there are multiple prompts)
     * idx.completion is the index of the completion that produced the token
     *   (if multiple completions per prompt are requested)
     */
    idx: NewTokenIndices,
    runId: string,
    parentRunId?: string,
    tags?: string[]
  ): Promise<void> | void;

  /**
   * Called if an LLM/ChatModel run encounters an error
   */
  handleLLMError?(
    err: Error,
    runId: string,
    parentRunId?: string,
    tags?: string[]
  ): Promise<void> | void;

  /**
   * Called at the end of an LLM/ChatModel run, with the output and the run ID.
   */
  handleLLMEnd?(
    output: LLMResult,
    runId: string,
    parentRunId?: string,
    tags?: string[]
  ): Promise<void> | void;

  /**
   * Called at the start of a Chat Model run, with the prompt(s)
   * and the run ID.
   */
  handleChatModelStart?(
    llm: Serialized,
    messages: BaseMessage[][],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    tags?: string[],
    metadata?: Record<string, unknown>
  ): Promise<void> | void;

  /**
   * Called at the start of a Chain run, with the chain name and inputs
   * and the run ID.
   */
  handleChainStart?(
    chain: Serialized,
    inputs: ChainValues,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Record<string, unknown>
  ): Promise<void> | void;

  /**
   * Called if a Chain run encounters an error
   */
  handleChainError?(
    err: Error,
    runId: string,
    parentRunId?: string,
    tags?: string[]
  ): Promise<void> | void;

  /**
   * Called at the end of a Chain run, with the outputs and the run ID.
   */
  handleChainEnd?(
    outputs: ChainValues,
    runId: string,
    parentRunId?: string,
    tags?: string[]
  ): Promise<void> | void;

  /**
   * Called at the start of a Tool run, with the tool name and input
   * and the run ID.
   */
  handleToolStart?(
    tool: Serialized,
    input: string,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Record<string, unknown>
  ): Promise<void> | void;

  /**
   * Called if a Tool run encounters an error
   */
  handleToolError?(
    err: Error,
    runId: string,
    parentRunId?: string,
    tags?: string[]
  ): Promise<void> | void;

  /**
   * Called at the end of a Tool run, with the tool output and the run ID.
   */
  handleToolEnd?(
    output: string,
    runId: string,
    parentRunId?: string,
    tags?: string[]
  ): Promise<void> | void;

  handleText?(
    text: string,
    runId: string,
    parentRunId?: string,
    tags?: string[]
  ): Promise<void> | void;

  /**
   * Called when an agent is about to execute an action,
   * with the action and the run ID.
   */
  handleAgentAction?(
    action: AgentAction,
    runId: string,
    parentRunId?: string,
    tags?: string[]
  ): Promise<void> | void;

  /**
   * Called when an agent finishes execution, before it exits.
   * with the final output and the run ID.
   */
  handleAgentEnd?(
    action: AgentFinish,
    runId: string,
    parentRunId?: string,
    tags?: string[]
  ): Promise<void> | void;

  /**
   * Called when embedding throws an error, after call to embedding.
   * with the error and the run ID
   */
  handleEmbeddingError?(
    err: Error,
    runId: string,
    parentRunId?: string
  ): Promise<void> | void;

  /**
   * Called when embedding is started, before call to embedding.
   * with the input text and the run ID
   */
  handleEmbeddingStart?(
    embedding: Serialized,
    texts: string[],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    tags?: string[],
    metadata?: Record<string, unknown>
  ): Promise<void> | void;

  /**
   * Called after embedding is completed, before it exits.
   * with embeddings vector and the run ID
   */
  handleEmbeddingEnd?(
    vector: number[],
    runId: string,
    parentRunId?: string
  ): Promise<void> | void;
}

/**
 * Base interface for callbacks. All methods are optional. If a method is not
 * implemented, it will be ignored. If a method is implemented, it will be
 * called at the appropriate time. All methods are called with the run ID of
 * the LLM/ChatModel/Chain that is running, which is generated by the
 * CallbackManager.
 *
 * @interface
 */
export type CallbackHandlerMethods = BaseCallbackHandlerMethodsClass;

export abstract class BaseCallbackHandler
  extends BaseCallbackHandlerMethodsClass
  implements BaseCallbackHandlerInput, Serializable
{
  lc_serializable = false;

  get lc_namespace(): ["langchain", "callbacks", string] {
    return ["langchain", "callbacks", this.name];
  }

  get lc_secrets(): { [key: string]: string } | undefined {
    return undefined;
  }

  get lc_attributes(): { [key: string]: string } | undefined {
    return undefined;
  }

  get lc_aliases(): { [key: string]: string } | undefined {
    return undefined;
  }

  lc_kwargs: SerializedFields;

  abstract name: string;

  ignoreLLM = false;

  ignoreChain = false;

  ignoreAgent = false;

  ignoreEmbeddings = false;

  awaitHandlers =
    typeof process !== "undefined"
      ? // eslint-disable-next-line no-process-env
        process.env?.LANGCHAIN_CALLBACKS_BACKGROUND !== "true"
      : true;

  constructor(input?: BaseCallbackHandlerInput) {
    super();
    this.lc_kwargs = input || {};
    if (input) {
      this.ignoreLLM = input.ignoreLLM ?? this.ignoreLLM;
      this.ignoreChain = input.ignoreChain ?? this.ignoreChain;
      this.ignoreAgent = input.ignoreAgent ?? this.ignoreAgent;
      this.ignoreEmbeddings = input.ignoreEmbeddings ?? this.ignoreEmbeddings;
    }
  }

  copy(): BaseCallbackHandler {
    return new (this.constructor as new (
      input?: BaseCallbackHandlerInput
    ) => BaseCallbackHandler)(this);
  }

  toJSON(): Serialized {
    return Serializable.prototype.toJSON.call(this);
  }

  toJSONNotImplemented(): SerializedNotImplemented {
    return Serializable.prototype.toJSONNotImplemented.call(this);
  }

  static fromMethods(methods: CallbackHandlerMethods) {
    class Handler extends BaseCallbackHandler {
      name = uuid.v4();

      constructor() {
        super();
        Object.assign(this, methods);
      }
    }
    return new Handler();
  }
}
