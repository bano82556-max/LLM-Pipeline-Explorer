/**
 * LLM Pipeline Explorer - Subword Tokenizer Engine
 * 
 * Implements a WordPiece-style subword tokenizer with:
 * - Corpus frequency analysis
 * - Dynamic vocabulary pruning via slider
 * - Greedy longest-match prefix & suffix (##) segmentation
 * - Detailed step-by-step tokenization trace generation
 */

class SubwordTokenizer {
  constructor() {
    // Special tokens (reserved IDs 0-3)
    this.specialTokens = [
      { token: "[PAD]", type: "special", desc: "Padding token for batching" },
      { token: "[UNK]", type: "special", desc: "Unknown token (unseen characters)" },
      { token: "[CLS]", type: "special", desc: "Classification token (start of sequence)" },
      { token: "[SEP]", type: "special", desc: "Separator token (end of sentence)" }
    ];

    // Common morphological subwords and affixes
    this.baseAffixes = [
      "##s", "##es", "##ed", "##ing", "##er", "##est", "##ly", "##y", 
      "##d", "##e", "##n", "##ny", "##ful", "##less", "##ment", "##able",
      "##ness", "##ish", "##al", "##ic", "##ity", "##tion"
    ];

    // Standard ASCII characters and punctuation
    this.baseChars = [];
    for (let i = 97; i <= 122; i++) { // a-z
      const ch = String.fromCharCode(i);
      this.baseChars.push(ch);
      this.baseChars.push("##" + ch);
    }
    for (let i = 48; i <= 57; i++) { // 0-9
      const digit = String.fromCharCode(i);
      this.baseChars.push(digit);
    }
    this.punctuation = [".", ",", "!", "?", "-", "'", "\"", ";", ":", "(", ")", "/", "@"];

    // Corpus state
    this.corpusWordFreq = {}; // { word: frequency }
    this.sortedCorpusWords = []; // [ [word, freq], ... ]
    this.activeVocabMap = new Map(); // token -> { id, token, type, freq, active }
    this.fullVocabList = []; // Complete list of all tokens
    this.wordsKeptCount = 0;
  }

  /**
   * Train/build vocabulary from raw training text
   * @param {string} trainingText 
   */
  train(trainingText) {
    this.corpusWordFreq = {};
    
    // Normalize and tokenize training text into words
    const rawTokens = trainingText.toLowerCase().match(/[a-z0-9]+|[^\s\w]/g) || [];
    
    for (const token of rawTokens) {
      if (/^[a-z0-9]+$/.test(token)) {
        this.corpusWordFreq[token] = (this.corpusWordFreq[token] || 0) + 1;
      }
    }

    // Sort words by frequency descending, then alphabetically for deterministic order
    this.sortedCorpusWords = Object.entries(this.corpusWordFreq).sort((a, b) => {
      if (b[1] !== a[1]) return b[1] - a[1];
      return a[0].localeCompare(b[0]);
    });

    // Default: keep all words
    this.wordsKeptCount = this.sortedCorpusWords.length;
    this.updateActiveVocabulary(this.wordsKeptCount);
  }

  /**
   * Rebuild the active vocabulary given slider value k (words kept)
   * @param {number} k Number of whole words to keep in vocabulary
   */
  updateActiveVocabulary(k) {
    this.wordsKeptCount = Math.max(0, Math.min(k, this.sortedCorpusWords.length));
    this.activeVocabMap.clear();
    this.fullVocabList = [];

    let currentId = 0;

    // 1. Special tokens
    for (const item of this.specialTokens) {
      const entry = {
        id: currentId++,
        token: item.token,
        type: "special",
        freq: "-",
        active: true,
        desc: item.desc
      };
      this.activeVocabMap.set(item.token, entry);
      this.fullVocabList.push(entry);
    }

    // 2. Base morphological affixes (##s, ##ed, etc.)
    for (const affix of this.baseAffixes) {
      const entry = {
        id: currentId++,
        token: affix,
        type: "subword_suffix",
        freq: "-",
        active: true,
        desc: "Subword piece (continuation suffix)"
      };
      this.activeVocabMap.set(affix, entry);
      this.fullVocabList.push(entry);
    }

    // 3. Characters & Punctuation
    for (const char of this.baseChars) {
      const isSub = char.startsWith("##");
      const entry = {
        id: currentId++,
        token: char,
        type: isSub ? "subword_char" : "base_char",
        freq: "-",
        active: true,
        desc: isSub ? "Character subword piece" : "Base character"
      };
      this.activeVocabMap.set(char, entry);
      this.fullVocabList.push(entry);
    }

    for (const p of this.punctuation) {
      const entry = {
        id: currentId++,
        token: p,
        type: "punctuation",
        freq: "-",
        active: true,
        desc: "Punctuation symbol"
      };
      this.activeVocabMap.set(p, entry);
      this.fullVocabList.push(entry);
    }

    // 4. Training corpus whole words
    // Top k words are ACTIVE; remaining are INACTIVE (pruned from vocabulary)
    this.sortedCorpusWords.forEach(([word, freq], idx) => {
      const isActive = idx < this.wordsKeptCount;
      const entry = {
        id: currentId++,
        token: word,
        type: "whole_word",
        freq: freq,
        active: isActive,
        rank: idx + 1,
        desc: isActive ? `Whole word (rank #${idx + 1}, freq ${freq})` : `Dropped from vocab (freq ${freq})`
      };
      if (isActive) {
        this.activeVocabMap.set(word, entry);
      }
      this.fullVocabList.push(entry);
    });

    return {
      activeCount: this.activeVocabMap.size,
      totalCount: this.fullVocabList.length,
      wordsKept: this.wordsKeptCount,
      totalWords: this.sortedCorpusWords.length
    };
  }

  /**
   * Tokenize a single word into active vocabulary tokens (WordPiece algorithm)
   * @param {string} rawWord 
   * @returns {{ tokens: Array, trace: Array }}
   */
  tokenizeWord(rawWord) {
    const word = rawWord.toLowerCase();
    const trace = [];

    // Check if the entire word exists in the active vocabulary
    if (this.activeVocabMap.has(word)) {
      const item = this.activeVocabMap.get(word);
      trace.push({
        step: `Matched entire word "${word}" as whole-word token (ID: ${item.id})`,
        success: true
      });
      return {
        tokens: [{
          token: word,
          id: item.id,
          type: "whole_word",
          original: rawWord
        }],
        trace
      };
    }

    // The whole word is NOT in vocabulary -> greedy subword decomposition!
    trace.push({
      step: `Word "${word}" is not in active vocabulary. Initiating subword decomposition...`,
      success: false
    });

    const subTokens = [];
    let start = 0;
    const len = word.length;

    while (start < len) {
      let end = len;
      let matchedToken = null;
      let matchedEntry = null;

      // Greedy longest-match search
      while (start < end) {
        let candidate = word.slice(start, end);
        if (start > 0) {
          candidate = "##" + candidate;
        }

        if (this.activeVocabMap.has(candidate)) {
          matchedToken = candidate;
          matchedEntry = this.activeVocabMap.get(candidate);
          break;
        }
        end--;
      }

      if (matchedToken) {
        subTokens.push({
          token: matchedToken,
          id: matchedEntry.id,
          type: matchedEntry.type,
          original: word.slice(start, end)
        });
        trace.push({
          step: `Matched subword "${matchedToken}" (ID: ${matchedEntry.id}, type: ${matchedEntry.type}) at index [${start}:${end}]`,
          success: true
        });
        start = end;
      } else {
        // No matching subword prefix could be found; fall back to single character
        const charCandidate = (start > 0 ? "##" : "") + word[start];
        if (this.activeVocabMap.has(charCandidate)) {
          const entry = this.activeVocabMap.get(charCandidate);
          subTokens.push({
            token: charCandidate,
            id: entry.id,
            type: entry.type,
            original: word[start]
          });
          trace.push({
            step: `Fallback to single character "${charCandidate}" (ID: ${entry.id})`,
            success: true
          });
        } else if (this.activeVocabMap.has(word[start])) {
          const entry = this.activeVocabMap.get(word[start]);
          subTokens.push({
            token: word[start],
            id: entry.id,
            type: entry.type,
            original: word[start]
          });
          trace.push({
            step: `Fallback to character "${word[start]}" (ID: ${entry.id})`,
            success: true
          });
        } else {
          // Character not recognized in vocabulary at all -> [UNK]
          const unkEntry = this.activeVocabMap.get("[UNK]");
          subTokens.push({
            token: "[UNK]",
            id: unkEntry ? unkEntry.id : 1,
            type: "unknown",
            original: word[start]
          });
          trace.push({
            step: `Unknown character "${word[start]}" mapped to [UNK]`,
            success: false
          });
        }
        start += 1;
      }
    }

    return { tokens: subTokens, trace };
  }

  /**
   * Tokenize an entire sentence
   * @param {string} sentence 
   * @returns {{ tokens: Array, traces: Array, stats: Object }}
   */
  tokenizeSentence(sentence) {
    if (!sentence || !sentence.trim()) {
      return { tokens: [], traces: [], stats: { total: 0, wholeWords: 0, subwords: 0, chars: 0, unks: 0 } };
    }

    // Split sentence into words and punctuation
    const matches = sentence.match(/[A-Za-z0-9]+|[^\s\w]/g) || [];
    const allTokens = [];
    const allTraces = [];

    for (const rawToken of matches) {
      if (/^[A-Za-z0-9]+$/.test(rawToken)) {
        const { tokens, trace } = this.tokenizeWord(rawToken);
        allTokens.push(...tokens);
        allTraces.push({ word: rawToken, trace });
      } else {
        // Punctuation or special symbols
        if (this.activeVocabMap.has(rawToken)) {
          const entry = this.activeVocabMap.get(rawToken);
          allTokens.push({
            token: rawToken,
            id: entry.id,
            type: "punctuation",
            original: rawToken
          });
        } else {
          const unkEntry = this.activeVocabMap.get("[UNK]");
          allTokens.push({
            token: "[UNK]",
            id: unkEntry ? unkEntry.id : 1,
            type: "unknown",
            original: rawToken
          });
        }
      }
    }

    // Calculate breakdown statistics
    const stats = {
      total: allTokens.length,
      wholeWords: allTokens.filter(t => t.type === "whole_word").length,
      subwords: allTokens.filter(t => t.type === "subword_suffix" || t.type === "subword_prefix").length,
      chars: allTokens.filter(t => t.type === "base_char" || t.type === "subword_char").length,
      puncts: allTokens.filter(t => t.type === "punctuation").length,
      unks: allTokens.filter(t => t.type === "unknown").length
    };

    return { tokens: allTokens, traces: allTraces, stats };
  }
}

// Export for browser & node test environments
if (typeof module !== "undefined" && module.exports) {
  module.exports = SubwordTokenizer;
}

