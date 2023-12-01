import { type IBaseProtocol } from "pg-promise";
import { PGVectorStore } from "langchain/vectorstores/pg";

/**
 * You can extend PGVectorStore and change how the query runs to
 * make it work with RLS. As an example, if you're using Supabase's
 * RLS feature, you can extend PGVectorStore like so:
 */

export class SupabasePGVectorStore extends PGVectorStore {
  jwt?: Record<string, any>;

  setJWT(jwt: Record<string, any>) {
    this.jwt = jwt;
  }

  unsetJWT() {
    this.jwt = undefined;
  }

  protected runQuery<R>(
    query: (t: IBaseProtocol<any>) => Promise<R>
  ): Promise<R> {
    return this.pgInstance.tx((t) => {
      const claimsSetting = "request.jwt.claims";
      const claims = JSON.stringify(this.jwt);
      return (
        t
          .query(`SELECT set_config($1, $2, true);`, [claimsSetting, claims])
          /**
           * You might want to use this.pgExtension.runQueryWrapper(t, query)
           * if you're using pg_embedding, since it needs to add
           * SET LOCAL enable_seqscan = off; in the query for some reason.
           * If you're using pgvector, you can just do
           * .then(() => query(t));
           */
          .then(() => this.pgExtension.runQueryWrapper(t, query))
      );
    });
  }
}