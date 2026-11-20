/**
 * LLM Pipeline Explorer - 3D Embeddings & PCA Engine
 * 
 * Provides:
 * - Semantic vector embedding generation (domain concepts + TF-IDF pooling)
 * - Exact Cosine Similarity calculation
 * - Principal Component Analysis (PCA) via Jacobi Eigendecomposition
 * - Real-time projection of custom user sentences into the learned 3D space
 */

class EmbeddingEngine {
  constructor() {
    // 24 Semantic Concept Dimensions
    this.dimensionNames = [
      "animal_pet", "feline_cat", "canine_dog", "cute_fluffy", "sleep_rest", "play_chase",
      "tech_code", "python_lang", "software_dev", "functions_classes", "bugs_debug", "algorithms",
      "finance_market", "stocks_shares", "investors_trade", "banking_rates", "economy_inflation", "volatility",
      "action_motion", "place_indoor", "place_outdoor", "nature_weather", "value_price", "positivity"
    ];
    this.dim = this.dimensionNames.length;

    // Comprehensive semantic lexicon
    this.lexicon = this.buildLexicon();

    // Model state
    this.baseSentences = [];
    this.baseVectors = [];
    this.meanVector = [];
    this.centeredVectors = [];
    this.eigenResults = [];
    this.coords3D = [];
    this.explainedVariance = [0, 0, 0];
  }

  /**
   * Builds the semantic lexicon covering animal, coding, and finance domains
   */
  buildLexicon() {
    const map = {};

    const register = (words, weights) => {
      const vec = new Array(this.dim).fill(0);
      for (const [dimIdx, val] of Object.entries(weights)) {
        vec[parseInt(dimIdx)] = val;
      }
      for (const w of words) {
        map[w.toLowerCase()] = vec;
      }
    };

    // Category 1: Animals & Pets
    register(["cat", "cats", "feline"], { 0: 0.9, 1: 0.98, 3: 0.7, 4: 0.5, 19: 0.8, 23: 0.6 });
    register(["kitten", "kittens", "kitty"], { 0: 0.95, 1: 0.98, 3: 0.95, 4: 0.6, 5: 0.5, 19: 0.8, 23: 0.9 });
    register(["dog", "dogs", "canine", "hound"], { 0: 0.92, 2: 0.98, 3: 0.7, 5: 0.8, 18: 0.6, 20: 0.7, 23: 0.7 });
    register(["puppy", "puppies", "pup"], { 0: 0.96, 2: 0.98, 3: 0.95, 5: 0.9, 18: 0.7, 20: 0.7, 23: 0.9 });
    register(["animal", "animals", "pet", "pets", "creature"], { 0: 0.95, 3: 0.5, 18: 0.4, 23: 0.5 });
    register(["fluffy", "furry", "soft"], { 0: 0.7, 3: 0.95, 4: 0.4, 19: 0.6, 23: 0.7 });
    register(["sleep", "sleeps", "sleeping", "slept", "nap", "rest", "peaceful", "peacefully", "quiet", "quietly"], { 0: 0.3, 4: 0.98, 18: -0.5, 19: 0.8, 23: 0.7 });
    register(["play", "plays", "playing", "playful"], { 0: 0.6, 5: 0.95, 18: 0.8, 20: 0.7, 23: 0.85 });
    register(["chase", "chases", "chasing", "fetch", "catch"], { 0: 0.5, 5: 0.95, 18: 0.95, 20: 0.8, 23: 0.4 });
    register(["ball", "tennis", "toy"], { 0: 0.4, 5: 0.8, 18: 0.5, 20: 0.6 });
    register(["rug", "mat", "carpet", "blanket", "couch", "bed"], { 0: 0.3, 4: 0.8, 19: 0.95, 23: 0.4 });
    register(["cuddle", "cuddles", "cuddling", "hug"], { 0: 0.7, 1: 0.5, 2: 0.5, 3: 0.85, 4: 0.7, 19: 0.9, 23: 0.95 });
    register(["yard", "garden", "lawn", "park", "grass"], { 0: 0.4, 5: 0.5, 20: 0.95, 21: 0.8 });
    register(["warm", "sunny", "sunshine", "sun"], { 0: 0.2, 4: 0.5, 20: 0.6, 21: 0.95, 23: 0.8 });

    // Category 2: Programming & Computer Science
    register(["python", "py"], { 6: 0.98, 7: 0.98, 8: 0.85, 9: 0.8, 10: 0.7, 11: 0.8, 19: 0.6 });
    register(["programming", "coding", "programmer", "coder"], { 6: 0.98, 8: 0.95, 9: 0.9, 10: 0.8, 11: 0.85 });
    register(["code", "script", "scripts", "source"], { 6: 0.98, 7: 0.7, 8: 0.9, 9: 0.9, 10: 0.8, 11: 0.8 });
    register(["software", "application", "app", "system", "program"], { 6: 0.95, 8: 0.98, 9: 0.8, 10: 0.7, 11: 0.75 });
    register(["developer", "developers", "engineer", "engineers"], { 6: 0.92, 8: 0.95, 9: 0.8, 10: 0.8, 11: 0.8 });
    register(["function", "functions", "method", "methods"], { 6: 0.85, 7: 0.7, 8: 0.85, 9: 0.98, 11: 0.85 });
    register(["class", "classes", "object", "objects", "oop"], { 6: 0.85, 7: 0.7, 8: 0.85, 9: 0.98, 11: 0.8 });
    register(["loop", "loops", "iteration", "recursion"], { 6: 0.85, 8: 0.8, 9: 0.9, 11: 0.9 });
    register(["bug", "bugs", "error", "errors", "defect", "fault"], { 6: 0.8, 8: 0.8, 10: 0.98, 17: 0.5, 23: -0.5 });
    register(["debug", "debugging", "debugger", "fix", "fixed"], { 6: 0.88, 8: 0.85, 10: 0.98, 18: 0.5, 23: 0.5 });
    register(["algorithm", "algorithms", "logic", "computational"], { 6: 0.88, 7: 0.6, 8: 0.8, 11: 0.98 });
    register(["deploy", "deployment", "deployed", "devops"], { 6: 0.85, 8: 0.95, 18: 0.6 });
    register(["backend", "frontend", "fullstack", "api", "apis", "server"], { 6: 0.9, 8: 0.95, 9: 0.7 });
    register(["clean", "versatile", "efficient", "optimal"], { 6: 0.5, 8: 0.6, 9: 0.7, 23: 0.8 });

    // Category 3: Finance & Stock Market
    register(["stock", "stocks", "share", "shares", "equity", "equities"], { 12: 0.98, 13: 0.98, 14: 0.9, 17: 0.6, 22: 0.95 });
    register(["market", "markets", "bourse", "nasdaq", "dow", "sp500", "indices", "index"], { 12: 0.98, 13: 0.85, 14: 0.85, 15: 0.7, 16: 0.8, 17: 0.7, 22: 0.9 });
    register(["wall", "street"], { 12: 0.95, 13: 0.9, 14: 0.9, 15: 0.7, 22: 0.9 });
    register(["investor", "investors", "trader", "traders", "fund", "funds"], { 12: 0.92, 13: 0.85, 14: 0.98, 17: 0.6, 22: 0.9 });
    register(["trade", "trades", "trading", "traded", "buy", "sell"], { 12: 0.9, 13: 0.8, 14: 0.95, 17: 0.65, 18: 0.6, 22: 0.9 });
    register(["bond", "bonds", "treasury", "yield", "yields"], { 12: 0.85, 13: 0.7, 14: 0.8, 15: 0.95, 16: 0.7, 22: 0.9 });
    register(["bank", "banking", "central", "fed", "federal", "reserve"], { 12: 0.85, 14: 0.7, 15: 0.98, 16: 0.8, 22: 0.85 });
    register(["rate", "rates", "interest"], { 12: 0.8, 15: 0.98, 16: 0.85, 22: 0.8 });
    register(["inflation", "deflation", "cpi"], { 12: 0.8, 15: 0.8, 16: 0.98, 17: 0.8, 22: 0.85, 23: -0.6 });
    register(["volatility", "volatile", "fluctuation", "risk"], { 12: 0.85, 13: 0.75, 14: 0.7, 17: 0.98, 23: -0.5 });
    register(["rally", "rallied", "bullish", "gain", "gains", "surge"], { 12: 0.88, 13: 0.9, 17: 0.7, 18: 0.7, 22: 0.9, 23: 0.8 });
    register(["economic", "economy", "recession", "gdp"], { 12: 0.9, 15: 0.85, 16: 0.95, 22: 0.85 });

    return map;
  }

  /**
   * Retrieves or generates a vector for an individual word
   */
  getWordVector(word) {
    const w = word.toLowerCase().trim();
    if (!w) return new Array(this.dim).fill(0);

    // 1. Direct lexicon hit
    if (this.lexicon[w]) return this.lexicon[w];

    // 2. Stemming / Prefix matching
    for (const [key, vec] of Object.entries(this.lexicon)) {
      if (w.startsWith(key) || key.startsWith(w)) {
        return vec;
      }
    }

    // 3. Fallback: deterministic character n-gram hash projection
    let hash = 0;
    for (let i = 0; i < w.length; i++) {
      hash = (hash * 31 + w.charCodeAt(i)) & 0xffffffff;
    }
    const pseudoVec = [];
    for (let d = 0; d < this.dim; d++) {
      pseudoVec.push(Math.sin(hash + d * 1.618) * 0.1);
    }
    return pseudoVec;
  }

  /**
   * Embeds an entire sentence into a 24D unit vector
   */
  embedSentence(text) {
    const stopWords = new Set([
      "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", 
      "of", "with", "by", "its", "their", "is", "are", "was", "were", 
      "it", "this", "that", "these", "those", "be", "been", "as"
    ]);

    const words = (text || "").toLowerCase().match(/[a-z0-9]+/g) || [];
    const sentenceVec = new Array(this.dim).fill(0);
    let totalWeight = 0;

    for (const word of words) {
      if (stopWords.has(word)) continue;
      const wordVec = this.getWordVector(word);
      const weight = 1.0; // Uniform weight for content words
      for (let d = 0; d < this.dim; d++) {
        sentenceVec[d] += wordVec[d] * weight;
      }
      totalWeight += weight;
    }

    if (totalWeight === 0) totalWeight = 1.0;
    for (let d = 0; d < this.dim; d++) {
      sentenceVec[d] /= totalWeight;
    }

    // Normalize to unit norm: ||v|| = 1
    const norm = Math.hypot(...sentenceVec);
    if (norm > 1e-9) {
      for (let d = 0; d < this.dim; d++) {
        sentenceVec[d] /= norm;
      }
    }

    return sentenceVec;
  }

  /**
   * Calculates cosine similarity between two vectors
   */
  cosineSimilarity(vecA, vecB) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < vecA.length; i++) {
      dot += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    if (denom < 1e-9) return 0;
    return Math.max(-1, Math.min(1, dot / denom));
  }

  /**
   * Performs Jacobi Eigendecomposition on an NxN symmetric matrix
   */
  jacobiEigenvalues(A, maxIter = 100) {
    const n = A.length;
    let V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 1 : 0));
    let D = A.map(row => [...row]);

    for (let iter = 0; iter < maxIter; iter++) {
      let maxOff = 0, p = 0, q = 1;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (Math.abs(D[i][j]) > maxOff) {
            maxOff = Math.abs(D[i][j]);
            p = i;
            q = j;
          }
        }
      }
      if (maxOff < 1e-9) break;

      const app = D[p][p], aqq = D[q][q], apq = D[p][q];
      const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
      const c = Math.cos(phi), s = Math.sin(phi);

      for (let i = 0; i < n; i++) {
        if (i !== p && i !== q) {
          const aip = D[i][p], aiq = D[i][q];
          D[i][p] = D[p][i] = c * aip - s * aiq;
          D[i][q] = D[q][i] = s * aip + c * aiq;
        }
      }
      D[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
      D[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
      D[p][q] = D[q][p] = 0;

      for (let i = 0; i < n; i++) {
        const vip = V[i][p], viq = V[i][q];
        V[i][p] = c * vip - s * viq;
        V[i][q] = s * vip + c * viq;
      }
    }

    const eigen = [];
    for (let i = 0; i < n; i++) {
      eigen.push({ val: Math.max(0, D[i][i]), vec: V.map(row => row[i]) });
    }
    eigen.sort((a, b) => b.val - a.val);
    return eigen;
  }

  /**
   * Fit PCA on an array of sentences and project them to 3D
   */
  fitAndProject(sentences) {
    this.baseSentences = sentences;
    this.baseVectors = sentences.map(s => this.embedSentence(s.text));
    const N = this.baseVectors.length;
    const D = this.dim;

    // 1. Calculate sample mean vector
    this.meanVector = new Array(D).fill(0);
    for (const v of this.baseVectors) {
      for (let d = 0; d < D; d++) {
        this.meanVector[d] += v[d] / N;
      }
    }

    // 2. Center vectors
    this.centeredVectors = this.baseVectors.map(v => 
      v.map((val, d) => val - this.meanVector[d])
    );

    // 3. Form Gram Matrix K = Xc * Xc^T (N x N)
    const K = Array.from({ length: N }, () => new Array(N).fill(0));
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        let dot = 0;
        for (let d = 0; d < D; d++) {
          dot += this.centeredVectors[i][d] * this.centeredVectors[j][d];
        }
        K[i][j] = dot;
      }
    }

    // 4. Jacobi Eigendecomposition
    this.eigenResults = this.jacobiEigenvalues(K);

    // 5. Compute Explained Variance Ratio for top 3 components
    const totalVariance = this.eigenResults.reduce((acc, e) => acc + e.val, 0) || 1;
    this.explainedVariance = [
      ((this.eigenResults[0]?.val || 0) / totalVariance) * 100,
      ((this.eigenResults[1]?.val || 0) / totalVariance) * 100,
      ((this.eigenResults[2]?.val || 0) / totalVariance) * 100
    ];

    // 6. Compute 3D Coordinates for base sentences
    this.coords3D = [];
    for (let i = 0; i < N; i++) {
      const x = (this.eigenResults[0]?.vec[i] || 0) * Math.sqrt(this.eigenResults[0]?.val || 0);
      const y = (this.eigenResults[1]?.vec[i] || 0) * Math.sqrt(this.eigenResults[1]?.val || 0);
      const z = (this.eigenResults[2]?.vec[i] || 0) * Math.sqrt(this.eigenResults[2]?.val || 0);
      this.coords3D.push({ x, y, z });
    }

    return {
      coords: this.coords3D,
      explainedVariance: this.explainedVariance
    };
  }

  /**
   * Project a new query sentence into the existing PCA 3D space
   */
  projectQuery(text) {
    const qVec = this.embedSentence(text);
    const qCentered = qVec.map((v, d) => v - this.meanVector[d]);
    const N = this.centeredVectors.length;
    const D = this.dim;

    // Project along top 3 principal axes: p_k = qCentered . w_k
    // where w_k = (1 / sqrt(lambda_k)) * Xc^T * u_k
    const coords = [0, 1, 2].map(pc => {
      const lambda = this.eigenResults[pc]?.val || 0;
      if (lambda < 1e-9) return 0;
      const u = this.eigenResults[pc].vec;
      let dotProduct = 0;
      for (let d = 0; d < D; d++) {
        let w_d = 0;
        for (let i = 0; i < N; i++) {
          w_d += this.centeredVectors[i][d] * u[i];
        }
        w_d /= Math.sqrt(lambda);
        dotProduct += qCentered[d] * w_d;
      }
      return dotProduct;
    });

    // Compute similarities against all existing sentences
    let maxSim = -2;
    let nearestIdx = -1;
    const similarities = this.baseVectors.map((bv, idx) => {
      const sim = this.cosineSimilarity(qVec, bv);
      if (sim > maxSim) {
        maxSim = sim;
        nearestIdx = idx;
      }
      return {
        sentenceIdx: idx,
        sentence: this.baseSentences[idx],
        similarity: sim
      };
    });

    return {
      vector: qVec,
      coords: { x: coords[0], y: coords[1], z: coords[2] },
      nearestIdx,
      nearestSentence: this.baseSentences[nearestIdx],
      maxSimilarity: maxSim,
      similarities
    };
  }

  /**
   * Computes top closest sentence pairs across the base sentences
   */
  getTopSimilarPairs(limit = 5) {
    const pairs = [];
    const N = this.baseVectors.length;
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const sim = this.cosineSimilarity(this.baseVectors[i], this.baseVectors[j]);
        pairs.push({
          sentA: this.baseSentences[i],
          sentB: this.baseSentences[j],
          idxA: i,
          idxB: j,
          similarity: sim
        });
      }
    }
    pairs.sort((a, b) => b.similarity - a.similarity);
    return pairs.slice(0, limit);
  }
}

// Export for browser & node test environments
if (typeof module !== "undefined" && module.exports) {
  module.exports = EmbeddingEngine;
}

