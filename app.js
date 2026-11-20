/**
 * LLM Pipeline Explorer - Main Application Controller
 * Orchestrates UI events, tokenizer engine, embedding engine, and Plotly 3D visualizations.
 */

// Global State
const appState = {
  tokenizer: new SubwordTokenizer(),
  embeddings: new EmbeddingEngine(),
  initialTokenCount: 0,
  currentFilter: 'all',
  searchQuery: '',
  sentencesData: [],
  userProjection: null,
  isRotating: false,
  rotationTimer: null,
  rotationAngle: 0
};

// Preset Corpus & Sentences
const DEFAULT_CORPUS = `The sunny days make the quick cats and dogs play happily.
A friendly dog likes to play with cats in the sunny garden.
The dogs liked playing outside in the warm sun.
The jumping cat jumps over the sleeping dog.
Sunny weather brings warm sun and happy cats.`;

const DEFAULT_SENTENCES = [
  { id: 1, text: "A fluffy cat sleeps peacefully on the warm rug.", group: "Pets & Animals" },
  { id: 2, text: "The playful puppy chases its favorite tennis ball.", group: "Pets & Animals" },
  { id: 3, text: "A cute kitten and puppy cuddle together on the mat.", group: "Pets & Animals" },
  { id: 4, text: "Python is a versatile programming language for software development.", group: "Programming & Code" },
  { id: 5, text: "Writing clean code with functions and classes prevents software bugs.", group: "Programming & Code" },
  { id: 6, text: "Developers build algorithms, debug scripts, and deploy backend APIs.", group: "Programming & Code" },
  { id: 7, text: "Stock market indices rallied sharply after central bank rate cuts.", group: "Finance & Markets" },
  { id: 8, text: "Wall Street investors trade equities, government bonds, and shares.", group: "Finance & Markets" },
  { id: 9, text: "Financial markets face high economic volatility and rising inflation.", group: "Finance & Markets" }
];

// Topic color definitions
const TOPIC_CONFIG = {
  "Pets & Animals": {
    color: "#2563eb", // Blue
    symbol: "circle",
    badgeBg: "bg-blue-50",
    badgeBorder: "border-blue-200",
    badgeText: "text-blue-700"
  },
  "Programming & Code": {
    color: "#059669", // Emerald
    symbol: "diamond",
    badgeBg: "bg-emerald-50",
    badgeBorder: "border-emerald-200",
    badgeText: "text-emerald-700"
  },
  "Finance & Markets": {
    color: "#7c3aed", // Violet/Purple
    symbol: "square",
    badgeBg: "bg-purple-50",
    badgeBorder: "border-purple-200",
    badgeText: "text-purple-700"
  },
  "User Input": {
    color: "#d97706", // Amber / Gold
    symbol: "cross",
    badgeBg: "bg-amber-50",
    badgeBorder: "border-amber-200",
    badgeText: "text-amber-800"
  }
};

// Initialize Application on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  // Initialize Lucide icons
  if (window.lucide) {
    window.lucide.createIcons();
  }

  initTokenizerSection();
  initEmbeddingsSection();
  initModals();
});

/* ==========================================================================
   SECTION 1: TOKENIZER INITIALIZATION & LOGIC
   ========================================================================== */

function initTokenizerSection() {
  const trainingInput = document.getElementById('trainingInput');
  const btnBuildVocab = document.getElementById('btnBuildVocab');
  const btnPresetCorpus = document.getElementById('btnPresetCorpus');
  const vocabSlider = document.getElementById('vocabSlider');
  const vocabSearch = document.getElementById('vocabSearch');
  const sentenceInput = document.getElementById('sentenceInput');
  const btnTokenize = document.getElementById('btnTokenize');
  const btnToggleTrace = document.getElementById('btnToggleTrace');

  // 1. Populate default training text
  trainingInput.value = DEFAULT_CORPUS;

  // 2. Build initial vocabulary
  buildVocabularyFromInput();

  // 3. Bind event listeners
  btnBuildVocab.addEventListener('click', () => {
    buildVocabularyFromInput();
    tokenizeCurrentSentence();
  });

  btnPresetCorpus.addEventListener('click', () => {
    trainingInput.value = DEFAULT_CORPUS;
    buildVocabularyFromInput();
    tokenizeCurrentSentence();
  });

  vocabSlider.addEventListener('input', (e) => {
    handleSliderChange(parseInt(e.target.value, 10));
  });

  vocabSearch.addEventListener('input', (e) => {
    appState.searchQuery = e.target.value.toLowerCase().trim();
    renderVocabTable();
  });

  // Filter chips for vocabulary table
  document.querySelectorAll('.vocab-filter-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.vocab-filter-btn').forEach(b => {
        b.className = 'vocab-filter-btn px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-600 hover:bg-slate-200';
      });
      e.target.className = 'vocab-filter-btn px-2 py-0.5 rounded text-xs font-medium bg-blue-600 text-white';
      appState.currentFilter = e.target.getAttribute('data-filter');
      renderVocabTable();
    });
  });

  btnTokenize.addEventListener('click', () => {
    tokenizeCurrentSentence();
  });

  sentenceInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      tokenizeCurrentSentence();
    }
  });

  // Preset test sentence buttons
  document.querySelectorAll('.test-sentence-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      sentenceInput.value = e.target.getAttribute('data-sentence');
      tokenizeCurrentSentence();
    });
  });

  // Toggle trace
  btnToggleTrace.addEventListener('click', () => {
    const traceContent = document.getElementById('traceContent');
    const chevron = document.getElementById('traceChevron');
    const isHidden = traceContent.classList.contains('hidden');
    if (isHidden) {
      traceContent.classList.remove('hidden');
      chevron.style.transform = 'rotate(180deg)';
    } else {
      traceContent.classList.add('hidden');
      chevron.style.transform = 'rotate(0deg)';
    }
  });

  // Perform initial sentence tokenization
  tokenizeCurrentSentence(true);
}

function buildVocabularyFromInput() {
  const text = document.getElementById('trainingInput').value;
  appState.tokenizer.train(text);

  const totalWords = appState.tokenizer.sortedCorpusWords.length;
  const slider = document.getElementById('vocabSlider');
  slider.max = totalWords;
  slider.value = totalWords;

  updateSliderUI(totalWords, totalWords);
  renderVocabTable();
}

function handleSliderChange(val) {
  const stats = appState.tokenizer.updateActiveVocabulary(val);
  updateSliderUI(val, stats.totalWords);
  renderVocabTable();
  tokenizeCurrentSentence(false);
}

function updateSliderUI(kept, total) {
  const badge = document.getElementById('sliderBadge');
  const pctText = document.getElementById('sliderPctText');
  const maxText = document.getElementById('sliderMaxText');
  const title = document.getElementById('sliderInsightTitle');
  const text = document.getElementById('sliderInsightText');

  const pct = total > 0 ? Math.round((kept / total) * 100) : 0;
  badge.textContent = `Words kept in vocabulary: ${kept} of ${total}`;
  pctText.textContent = `${pct}% capacity`;
  maxText.textContent = `${total} words`;

  if (kept === total) {
    title.textContent = "100% Full Vocabulary Capacity";
    text.innerHTML = `All <strong>${total}</strong> words from training text are in the vocabulary as whole words. Words tokenize into single tokens where possible!`;
  } else if (kept > total / 2) {
    title.textContent = `Moderate Vocabulary (${kept} of ${total} Words)`;
    text.innerHTML = `Dropped the rarest words from vocabulary. Notice how compound words decompose into root words and subword affixes (like <code class="text-emerald-700 font-mono">##s</code> or <code class="text-emerald-700 font-mono">##ed</code>)!`;
  } else if (kept > 0) {
    title.textContent = `Constrained Vocabulary (${kept} of ${total} Words)`;
    text.innerHTML = `Most whole words are evicted! The tokenizer relies heavily on morphological subwords and character pieces, significantly increasing the total token count.`;
  } else {
    title.textContent = "Minimal Vocabulary (0 Corpus Words Kept)";
    text.innerHTML = `No whole words kept! Words are completely broken down into single characters and subword pieces. Total token count is at its maximum!`;
  }
}

function renderVocabTable() {
  const tbody = document.getElementById('vocabTableBody');
  const totalCountEl = document.getElementById('vocabTotalCount');
  const activeCountEl = document.getElementById('vocabActiveCount');

  const list = appState.tokenizer.fullVocabList;
  totalCountEl.textContent = list.length;
  activeCountEl.textContent = appState.tokenizer.activeVocabMap.size;

  const query = appState.searchQuery;
  const filter = appState.currentFilter;

  // Filter items
  const filtered = list.filter(item => {
    // Type filter
    if (filter === 'whole_word' && item.type !== 'whole_word') return false;
    if (filter === 'subword' && item.type !== 'subword_suffix') return false;
    if (filter === 'char' && item.type !== 'base_char' && item.type !== 'subword_char') return false;
    if (filter === 'special' && item.type !== 'special') return false;

    // Search query
    if (query) {
      const matchToken = item.token.toLowerCase().includes(query);
      const matchId = String(item.id).includes(query);
      const matchType = item.type.toLowerCase().includes(query);
      return matchToken || matchId || matchType;
    }
    return true;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="py-6 text-center text-xs text-slate-400">No tokens found matching filter</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(item => {
    let typeBadge = '';
    if (item.type === 'whole_word') {
      typeBadge = '<span class="px-2 py-0.5 rounded text-[11px] font-medium bg-blue-50 text-blue-700 border border-blue-200">Whole Word</span>';
    } else if (item.type === 'subword_suffix') {
      typeBadge = '<span class="px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">Subword (##)</span>';
    } else if (item.type === 'base_char' || item.type === 'subword_char') {
      typeBadge = '<span class="px-2 py-0.5 rounded text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-200">Character</span>';
    } else if (item.type === 'special') {
      typeBadge = '<span class="px-2 py-0.5 rounded text-[11px] font-medium bg-purple-50 text-purple-700 border border-purple-200">Special</span>';
    } else {
      typeBadge = '<span class="px-2 py-0.5 rounded text-[11px] font-medium bg-slate-100 text-slate-600">Punctuation</span>';
    }

    const statusBadge = item.active 
      ? '<span class="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-100 text-emerald-800">Active</span>'
      : '<span class="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-500 line-through">Pruned</span>';

    const rowClass = item.active ? 'cursor-pointer' : 'opacity-50 cursor-not-allowed bg-slate-50/50';

    return `
      <tr class="${rowClass} transition-colors" onclick="selectVocabItem('${item.token.replace(/'/g, "\\'")}')">
        <td class="py-2 px-3 font-mono text-xs text-slate-500 font-semibold">#${item.id}</td>
        <td class="py-2 px-3 font-mono font-medium text-slate-900">${escapeHtml(item.token)}</td>
        <td class="py-2 px-3">${typeBadge}</td>
        <td class="py-2 px-3 text-right font-mono text-xs text-slate-600">${item.freq !== '-' ? item.freq : '&mdash;'}</td>
        <td class="py-2 px-3 text-center">${statusBadge}</td>
      </tr>
    `;
  }).join('');
}

window.selectVocabItem = function(token) {
  const clean = token.replace(/^##/, '');
  const input = document.getElementById('sentenceInput');
  input.value = `The ${clean} is here.`;
  tokenizeCurrentSentence();
};

function tokenizeCurrentSentence(isInitial = false) {
  const sentence = document.getElementById('sentenceInput').value;
  const result = appState.tokenizer.tokenizeSentence(sentence);

  if (isInitial) {
    appState.initialTokenCount = result.tokens.length;
  }

  // Update Metrics
  const totalTokensEl = document.getElementById('metricTotalTokens');
  const deltaEl = document.getElementById('metricTokensDelta');
  const wholeWordsEl = document.getElementById('metricWholeWords');
  const subwordsEl = document.getElementById('metricSubwords');
  const charsEl = document.getElementById('metricChars');
  const unksEl = document.getElementById('metricUnks');

  totalTokensEl.textContent = result.stats.total;
  wholeWordsEl.textContent = result.stats.wholeWords;
  subwordsEl.textContent = result.stats.subwords;
  charsEl.textContent = result.stats.chars;
  unksEl.textContent = result.stats.unks;

  // Delta indicator when slider changes token count
  if (!isInitial && appState.initialTokenCount > 0) {
    const diff = result.stats.total - appState.initialTokenCount;
    if (diff > 0) {
      deltaEl.textContent = `(+${diff} tokens from split)`;
      deltaEl.className = "text-xs font-semibold text-amber-600";
    } else if (diff < 0) {
      deltaEl.textContent = `(${diff} tokens)`;
      deltaEl.className = "text-xs font-semibold text-blue-600";
    } else {
      deltaEl.textContent = "";
    }
  } else {
    deltaEl.textContent = "";
  }

  // Render Token Pills
  renderTokenPills(result.tokens);

  // Render Decision Trace
  renderTrace(result.traces);
}

function renderTokenPills(tokens) {
  const container = document.getElementById('tokenPillsContainer');
  if (tokens.length === 0) {
    container.innerHTML = `<span class="text-xs text-slate-400 italic">Enter a sentence above and click "Tokenize"</span>`;
    return;
  }

  container.innerHTML = tokens.map(t => {
    let style = "";
    let typeTag = "";

    if (t.type === 'whole_word') {
      style = "bg-blue-50 border-blue-200 text-blue-900";
      typeTag = "Whole";
    } else if (t.type === 'subword_suffix') {
      style = "bg-emerald-50 border-emerald-300 text-emerald-900 font-semibold";
      typeTag = "Subword";
    } else if (t.type === 'base_char' || t.type === 'subword_char') {
      style = "bg-amber-50 border-amber-200 text-amber-900";
      typeTag = "Char";
    } else if (t.type === 'unknown') {
      style = "bg-rose-50 border-rose-300 text-rose-900";
      typeTag = "[UNK]";
    } else {
      style = "bg-slate-100 border-slate-300 text-slate-700";
      typeTag = "Punct";
    }

    return `
      <div class="token-pill flex flex-col items-center px-3 py-1.5 rounded-lg border shadow-xs ${style} cursor-pointer" title="Token: ${escapeHtml(t.token)} | ID: #${t.id} | Type: ${t.type}">
        <span class="font-mono text-sm font-semibold">${escapeHtml(t.token)}</span>
        <div class="flex items-center space-x-1 mt-0.5">
          <span class="text-[10px] font-mono opacity-70">#${t.id}</span>
          <span class="text-[9px] uppercase px-1 rounded bg-white/70 font-sans tracking-tight">${typeTag}</span>
        </div>
      </div>
    `;
  }).join('');
}

function renderTrace(traces) {
  const container = document.getElementById('traceContent');
  if (!traces || traces.length === 0) {
    container.innerHTML = `<p class="text-slate-400">No decision trace available</p>`;
    return;
  }

  container.innerHTML = traces.map(tr => {
    const stepsHtml = tr.trace.map(s => {
      const color = s.success ? "text-emerald-600" : "text-amber-600";
      return `<div class="pl-4 border-l-2 border-slate-200 py-0.5 ${color}">&rarr; ${escapeHtml(s.step)}</div>`;
    }).join('');

    return `
      <div class="p-2 bg-slate-50 rounded border border-slate-100 space-y-1">
        <span class="font-bold text-slate-900">Word: "${escapeHtml(tr.word)}"</span>
        ${stepsHtml}
      </div>
    `;
  }).join('');
}


/* ==========================================================================
   SECTION 2: 3D EMBEDDINGS & SEMANTIC SIMILARITY LOGIC
   ========================================================================== */

function initEmbeddingsSection() {
  const textarea = document.getElementById('sentencesTextarea');
  const btnEmbedPlot = document.getElementById('btnEmbedPlot');
  const btnResetCamera = document.getElementById('btnResetCamera');
  const btnToggleRotate = document.getElementById('btnToggleRotate');
  const userSentenceInput = document.getElementById('userSentenceInput');
  const btnProjectUserSentence = document.getElementById('btnProjectUserSentence');

  // 1. Populate default benchmark sentences
  textarea.value = DEFAULT_SENTENCES.map(s => s.text).join('\n');
  appState.sentencesData = [...DEFAULT_SENTENCES];

  // 2. Perform initial PCA fit & plot
  fitAndRenderPlot();

  // 3. Bind listeners
  btnEmbedPlot.addEventListener('click', () => {
    const lines = textarea.value.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length < 3) {
      alert("Please provide at least 3 sentences for 3D PCA projection!");
      return;
    }

    appState.sentencesData = lines.map((line, idx) => {
      // Determine topic based on index or keywords
      let group = "General Context";
      const lower = line.toLowerCase();
      if (lower.match(/cat|kitten|dog|puppy|rug|ball|mat|chase|sleep/)) {
        group = "Pets & Animals";
      } else if (lower.match(/python|code|programming|software|function|class|bug|algorithm|api/)) {
        group = "Programming & Code";
      } else if (lower.match(/stock|market|investor|trade|share|bond|rate|inflation|volatility|wall/)) {
        group = "Finance & Markets";
      } else {
        group = idx < 3 ? "Pets & Animals" : (idx < 6 ? "Programming & Code" : "Finance & Markets");
      }
      return { id: idx + 1, text: line, group };
    });

    appState.userProjection = null;
    document.getElementById('userProjectionResult').classList.add('hidden');
    fitAndRenderPlot();
  });

  btnResetCamera.addEventListener('click', () => {
    resetPlotCamera();
  });

  btnToggleRotate.addEventListener('click', () => {
    togglePlotAutoRotate();
  });

  btnProjectUserSentence.addEventListener('click', () => {
    handleUserSentenceProjection();
  });

  userSentenceInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      handleUserSentenceProjection();
    }
  });

  // Query chips
  document.querySelectorAll('.custom-query-chip').forEach(chip => {
    chip.addEventListener('click', (e) => {
      userSentenceInput.value = e.target.getAttribute('data-query');
      handleUserSentenceProjection();
    });
  });
}

function fitAndRenderPlot() {
  const { coords, explainedVariance } = appState.embeddings.fitAndProject(appState.sentencesData);

  // Update Explained Variance text
  const varEl = document.getElementById('pcaVarianceText');
  varEl.textContent = `PC1: ${explainedVariance[0].toFixed(1)}% | PC2: ${explainedVariance[1].toFixed(1)}% | PC3: ${explainedVariance[2].toFixed(1)}%`;

  // Update Closest Pairs Table
  renderClosestPairsTable();

  // Render 3D Plot with Plotly
  renderPlotly3D(coords);
}

function renderPlotly3D(coords) {
  const container = document.getElementById('plot3DContainer');
  const sentences = appState.sentencesData;

  // Group sentences by topic to create distinct Plotly traces
  const groups = {};
  sentences.forEach((s, idx) => {
    if (!groups[s.group]) groups[s.group] = [];
    groups[s.group].push({
      ...s,
      coord: coords[idx]
    });
  });

  const traces = [];

  for (const [groupName, items] of Object.entries(groups)) {
    const cfg = TOPIC_CONFIG[groupName] || { color: "#64748b", symbol: "circle" };

    traces.push({
      type: 'scatter3d',
      mode: 'markers+text',
      name: groupName,
      x: items.map(i => i.coord.x),
      y: items.map(i => i.coord.y),
      z: items.map(i => i.coord.z),
      text: items.map(i => `#${i.id}`),
      textposition: 'top center',
      textfont: {
        family: 'Inter, sans-serif',
        size: 11,
        color: '#1e293b'
      },
      hovertext: items.map(i => `<b>#${i.id} [${groupName}]</b><br>"${escapeHtml(i.text)}"<br><span style="font-size:10px;color:#94a3b8">(${i.coord.x.toFixed(2)}, ${i.coord.y.toFixed(2)}, ${i.coord.z.toFixed(2)})</span>`),
      hoverinfo: 'text',
      marker: {
        size: 8,
        color: cfg.color,
        symbol: cfg.symbol,
        opacity: 0.9,
        line: {
          color: '#ffffff',
          width: 1.5
        }
      }
    });
  }

  // If user has projected a custom query, add it as a highlight trace!
  if (appState.userProjection) {
    const up = appState.userProjection;
    const nearest = up.nearestSentence;
    const nearestCoord = appState.embeddings.coords3D[up.nearestIdx];

    // 1. User marker
    traces.push({
      type: 'scatter3d',
      mode: 'markers+text',
      name: 'Your Input',
      x: [up.coords.x],
      y: [up.coords.y],
      z: [up.coords.z],
      text: ['★ You'],
      textposition: 'top center',
      textfont: {
        family: 'Inter, sans-serif',
        size: 13,
        color: '#b45309'
      },
      hovertext: [`<b>Your Input (Projected)</b><br>"${escapeHtml(up.text)}"<br>Closest: #${nearest.id} (Sim: ${up.maxSimilarity.toFixed(3)})`],
      hoverinfo: 'text',
      marker: {
        size: 14,
        color: '#f59e0b',
        symbol: 'diamond',
        line: {
          color: '#ffffff',
          width: 2.5
        }
      }
    });

    // 2. Dashed 3D line connecting to nearest semantic neighbor
    if (nearestCoord) {
      traces.push({
        type: 'scatter3d',
        mode: 'lines',
        name: `Nearest Match (#${nearest.id})`,
        x: [up.coords.x, nearestCoord.x],
        y: [up.coords.y, nearestCoord.y],
        z: [up.coords.z, nearestCoord.z],
        line: {
          color: '#f59e0b',
          width: 3,
          dash: 'dash'
        },
        hoverinfo: 'none',
        showlegend: false
      });
    }
  }

  const layout = {
    margin: { l: 0, r: 0, b: 0, t: 0 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    showlegend: true,
    legend: {
      x: 0,
      y: 1,
      bgcolor: 'rgba(255, 255, 255, 0.85)',
      bordercolor: '#e2e8f0',
      borderwidth: 1,
      font: { size: 11, family: 'Inter, sans-serif' }
    },
    scene: {
      xaxis: {
        title: { text: 'PC 1: Primary Axis', font: { size: 10, color: '#64748b' } },
        backgroundcolor: '#f8fafc',
        gridcolor: '#e2e8f0',
        showbackground: true,
        zerolinecolor: '#cbd5e1'
      },
      yaxis: {
        title: { text: 'PC 2: Context Axis', font: { size: 10, color: '#64748b' } },
        backgroundcolor: '#f8fafc',
        gridcolor: '#e2e8f0',
        showbackground: true,
        zerolinecolor: '#cbd5e1'
      },
      zaxis: {
        title: { text: 'PC 3: Nuance Axis', font: { size: 10, color: '#64748b' } },
        backgroundcolor: '#f8fafc',
        gridcolor: '#e2e8f0',
        showbackground: true,
        zerolinecolor: '#cbd5e1'
      },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.2 }
      }
    }
  };

  const config = {
    responsive: true,
    displayModeBar: false
  };

  Plotly.react(container, traces, layout, config);
}

function handleUserSentenceProjection() {
  const input = document.getElementById('userSentenceInput');
  const text = input.value.trim();
  if (!text) return;

  // 1. Run through Section 1 tokenizer to showcase end-to-end integration
  const tokenRes = appState.tokenizer.tokenizeSentence(text);
  const userTokenPills = document.getElementById('userTokenPills');
  userTokenPills.innerHTML = tokenRes.tokens.map(t => `
    <span class="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono bg-amber-50 text-amber-800 border border-amber-200">
      ${escapeHtml(t.token)} (#${t.id})
    </span>
  `).join('');

  // 2. Project into PCA 3D space
  const proj = appState.embeddings.projectQuery(text);
  proj.text = text;
  appState.userProjection = proj;

  // 3. Update Result Card
  const resultCard = document.getElementById('userProjectionResult');
  resultCard.classList.remove('hidden');

  const coordsBadge = document.getElementById('userCoordsBadge');
  coordsBadge.textContent = `(x: ${proj.coords.x.toFixed(2)}, y: ${proj.coords.y.toFixed(2)}, z: ${proj.coords.z.toFixed(2)})`;

  const nearestText = document.getElementById('userNearestText');
  nearestText.textContent = `"#${proj.nearestSentence.id}: ${proj.nearestSentence.text}"`;

  const nearestScore = document.getElementById('userNearestScore');
  nearestScore.textContent = proj.maxSimilarity.toFixed(3);

  // 4. Re-render 3D Plot with user highlight
  renderPlotly3D(appState.embeddings.coords3D);
}

function renderClosestPairsTable() {
  const tbody = document.getElementById('closestPairsTableBody');
  const pairs = appState.embeddings.getTopSimilarPairs(5);

  tbody.innerHTML = pairs.map((p, idx) => {
    const cfgA = TOPIC_CONFIG[p.sentA.group] || TOPIC_CONFIG["Pets & Animals"];
    const cfgB = TOPIC_CONFIG[p.sentB.group] || TOPIC_CONFIG["Pets & Animals"];

    // Similarity score bar width (0.0 to 1.0)
    const pct = Math.max(0, Math.min(100, Math.round(p.similarity * 100)));

    return `
      <tr>
        <td class="py-2.5 px-3">
          <div class="flex items-center space-x-1.5">
            <span class="w-1.5 h-1.5 rounded-full" style="background-color: ${cfgA.color}"></span>
            <span class="text-xs font-medium text-slate-800 line-clamp-1" title="${escapeHtml(p.sentA.text)}">
              #${p.sentA.id} ${escapeHtml(p.sentA.text.substring(0, 26))}...
            </span>
          </div>
        </td>
        <td class="py-2.5 px-3">
          <div class="flex items-center space-x-1.5">
            <span class="w-1.5 h-1.5 rounded-full" style="background-color: ${cfgB.color}"></span>
            <span class="text-xs font-medium text-slate-800 line-clamp-1" title="${escapeHtml(p.sentB.text)}">
              #${p.sentB.id} ${escapeHtml(p.sentB.text.substring(0, 26))}...
            </span>
          </div>
        </td>
        <td class="py-2.5 px-3 text-right">
          <div class="flex flex-col items-end">
            <span class="font-mono text-xs font-bold text-emerald-700">${p.similarity.toFixed(3)}</span>
            <div class="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden mt-1">
              <div class="h-full bg-emerald-500 rounded-full" style="width: ${pct}%"></div>
            </div>
          </div>
        </td>
      </tr>
    `;
  }).join('');
}

function resetPlotCamera() {
  const container = document.getElementById('plot3DContainer');
  Plotly.relayout(container, {
    'scene.camera.eye': { x: 1.5, y: 1.5, z: 1.2 }
  });
}

function togglePlotAutoRotate() {
  const btnLabel = document.getElementById('rotateBtnLabel');
  const container = document.getElementById('plot3DContainer');

  if (appState.isRotating) {
    // Stop rotating
    appState.isRotating = false;
    btnLabel.textContent = "Auto-Rotate";
    if (appState.rotationTimer) {
      clearInterval(appState.rotationTimer);
      appState.rotationTimer = null;
    }
  } else {
    // Start rotating
    appState.isRotating = true;
    btnLabel.textContent = "Pause Rotation";
    const radius = 2.0;

    appState.rotationTimer = setInterval(() => {
      appState.rotationAngle += 0.02;
      const x = radius * Math.cos(appState.rotationAngle);
      const y = radius * Math.sin(appState.rotationAngle);
      Plotly.relayout(container, {
        'scene.camera.eye': { x: x, y: y, z: 1.2 }
      });
    }, 40);
  }
}


/* ==========================================================================
   MODALS: CONCEPT GUIDE & FULL MATRIX HEATMAP
   ========================================================================== */

function initModals() {
  // Guide Modal
  const modalGuide = document.getElementById('modalGuide');
  const btnOpenGuide = document.getElementById('btnOpenGuide');
  const btnCloseGuide = document.getElementById('btnCloseGuide');
  const btnCloseGuideBtn = document.getElementById('btnCloseGuideBtn');

  btnOpenGuide.addEventListener('click', () => modalGuide.classList.remove('hidden'));
  btnCloseGuide.addEventListener('click', () => modalGuide.classList.add('hidden'));
  btnCloseGuideBtn.addEventListener('click', () => modalGuide.classList.add('hidden'));

  // Matrix Modal
  const modalMatrix = document.getElementById('modalMatrix');
  const btnOpenMatrix = document.getElementById('btnOpenMatrix');
  const btnCloseMatrix = document.getElementById('btnCloseMatrix');
  const btnCloseMatrixBtn = document.getElementById('btnCloseMatrixBtn');

  btnOpenMatrix.addEventListener('click', () => {
    renderFullMatrix();
    modalMatrix.classList.remove('hidden');
  });
  btnCloseMatrix.addEventListener('click', () => modalMatrix.classList.add('hidden'));
  btnCloseMatrixBtn.addEventListener('click', () => modalMatrix.classList.add('hidden'));

  // Close modals on outside click
  window.addEventListener('click', (e) => {
    if (e.target === modalGuide) modalGuide.classList.add('hidden');
    if (e.target === modalMatrix) modalMatrix.classList.add('hidden');
  });
}

function renderFullMatrix() {
  const table = document.getElementById('matrixTable');
  const sentences = appState.sentencesData;
  const vectors = appState.embeddings.baseVectors;
  const N = sentences.length;

  let html = '<thead><tr><th class="py-2 px-3 text-left">Sentence</th>';
  for (let j = 0; j < N; j++) {
    html += `<th class="py-2 px-2 text-center font-mono">#${sentences[j].id}</th>`;
  }
  html += '</tr></thead><tbody>';

  for (let i = 0; i < N; i++) {
    html += `<tr><td class="py-2 px-3 text-left font-medium text-slate-800 text-xs">#${sentences[i].id} (${sentences[i].group})</td>`;
    for (let j = 0; j < N; j++) {
      const sim = appState.embeddings.cosineSimilarity(vectors[i], vectors[j]);
      let cellBg = "bg-white text-slate-400";
      if (i === j) {
        cellBg = "bg-slate-100 font-bold text-slate-900";
      } else if (sim >= 0.90) {
        cellBg = "bg-emerald-100 font-bold text-emerald-800";
      } else if (sim >= 0.60) {
        cellBg = "bg-emerald-50 text-emerald-700";
      } else if (sim >= 0.30) {
        cellBg = "bg-blue-50 text-blue-700";
      }
      html += `<td class="py-2 px-2 font-mono text-xs ${cellBg}">${sim.toFixed(3)}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody>';
  table.innerHTML = html;
}

// Utility: escape HTML
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

