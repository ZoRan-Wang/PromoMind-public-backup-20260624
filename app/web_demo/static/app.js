const state = {
  portfolios: [],
  households: [],
  portfoliosByHousehold: {},
  presets: [],
  metrics: null,
};

const COUPON_SLOTS = 10;

const el = {
  householdSelect: document.querySelector("#householdSelect"),
  timeSlider: document.querySelector("#timeSlider"),
  timeWindowValue: document.querySelector("#timeWindowValue"),
  timeWindowMeta: document.querySelector("#timeWindowMeta"),
  timeTicks: document.querySelector("#timeTicks"),
  presetButtons: document.querySelector("#presetButtons"),
  metricRecall: document.querySelector("#metricRecall"),
  metricNdcg: document.querySelector("#metricNdcg"),
  metricHit: document.querySelector("#metricHit"),
  metricLateExact: document.querySelector("#metricLateExact"),
  metricLateCategory: document.querySelector("#metricLateCategory"),
  eventTitle: document.querySelector("#eventTitle"),
  campaignType: document.querySelector("#campaignType"),
  modelName: document.querySelector("#modelName"),
  couponStart: document.querySelector("#couponStart"),
  predictedPurchase: document.querySelector("#predictedPurchase"),
  top10Hits: document.querySelector("#top10Hits"),
  eligibleSlots: document.querySelector("#eligibleSlots"),
  storyTag: document.querySelector("#storyTag"),
  storyText: document.querySelector("#storyText"),
  avgScore: document.querySelector("#avgScore"),
  historyMeta: document.querySelector("#historyMeta"),
  lateEvidenceMeta: document.querySelector("#lateEvidenceMeta"),
  lateEvidenceCards: document.querySelector("#lateEvidenceCards"),
  recommendationRows: document.querySelector("#recommendationRows"),
  historyRows: document.querySelector("#historyRows"),
  metricBars: document.querySelector("#metricBars"),
};

function pct(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function score(value) {
  return Number(value).toFixed(4);
}

function text(node, value) {
  node.textContent = value;
}

function child(tag, className, value) {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (value !== undefined) {
    node.textContent = value;
  }
  return node;
}

function setBootstrap(data) {
  state.portfolios = data.portfolios;
  state.presets = data.presets;
  state.metrics = data.metrics;
  buildHouseholdGroups();
  renderHouseholdOptions();
  renderTimeControl();
  renderPresets();
  renderHeadlineMetrics();
  renderMetricBars();
  loadSelectedPortfolio();
}

function buildHouseholdGroups() {
  state.portfoliosByHousehold = {};
  state.portfolios.forEach((portfolio) => {
    const householdId = String(portfolio.household_id);
    if (!state.portfoliosByHousehold[householdId]) {
      state.portfoliosByHousehold[householdId] = [];
    }
    state.portfoliosByHousehold[householdId].push(portfolio);
  });

  state.households = Object.entries(state.portfoliosByHousehold).map(([householdId, windows]) => {
    windows.sort((a, b) => a.coupon_start_date.localeCompare(b.coupon_start_date));
    const bestIndex = bestWindowIndex(windows);
    return {
      household_id: householdId,
      windows,
      best_index: bestIndex,
      best: windows[bestIndex],
    };
  });

  state.households.sort(
    (a, b) =>
      Number(b.windows.length > 1) - Number(a.windows.length > 1) ||
      comparePortfolio(b.best, a.best) ||
      Number(a.household_id) - Number(b.household_id),
  );
}

function bestWindowIndex(windows) {
  let bestIndex = 0;
  windows.forEach((portfolio, index) => {
    if (comparePortfolio(portfolio, windows[bestIndex]) > 0) {
      bestIndex = index;
    }
  });
  return bestIndex;
}

function comparePortfolio(a, b) {
  return (
    Number(a.positive_in_top10 >= 2) - Number(b.positive_in_top10 >= 2) ||
    Number(a.positive_in_top10) - Number(b.positive_in_top10) ||
    Number(a.late_exact_product) - Number(b.late_exact_product) ||
    Number(a.late_same_category) - Number(b.late_same_category)
  );
}

function renderHouseholdOptions() {
  el.householdSelect.replaceChildren();
  state.households.forEach((household) => {
    const option = document.createElement("option");
    const best = household.best;
    const prefix = portfolioSignal(best);
    option.value = household.household_id;
    option.textContent = `HH ${household.household_id} - ${household.windows.length} windows - ${prefix}`;
    el.householdSelect.append(option);
  });
  el.householdSelect.value = firstPresetHouseholdId();
}

function selectedHousehold() {
  return state.households.find((household) => household.household_id === el.householdSelect.value);
}

function selectedPortfolio() {
  const household = selectedHousehold();
  return household.windows[Number(el.timeSlider.value)];
}

function renderTimeControl(portfolioId) {
  const household = selectedHousehold();
  let index = household.best_index;
  if (portfolioId) {
    const matchedIndex = household.windows.findIndex((portfolio) => portfolio.portfolio_id === portfolioId);
    index = matchedIndex >= 0 ? matchedIndex : index;
  }

  el.timeSlider.min = 0;
  el.timeSlider.max = household.windows.length - 1;
  el.timeSlider.value = index;
  el.timeSlider.disabled = household.windows.length === 1;
  renderTimeTicks(household.windows);
  updateTimeReadout(household.windows[index], index, household.windows.length);
}

function renderTimeTicks(windows) {
  el.timeTicks.replaceChildren();
  windows.forEach((portfolio, index) => {
    const tick = child("span", "", portfolio.coupon_start_date.slice(5));
    tick.style.setProperty("--x", windows.length === 1 ? 0 : index / (windows.length - 1));
    el.timeTicks.append(tick);
  });
}

function updateTimeReadout(portfolio, index, total) {
  text(el.timeWindowValue, portfolio.coupon_start_date);
  text(el.timeWindowMeta, `${index + 1} of ${total} windows - ${portfolioSignal(portfolio)}`);
}

function portfolioSignal(portfolio) {
  if (portfolio.positive_in_top10 > 0) {
    return `HIT ${portfolio.positive_in_top10}`;
  }
  if (portfolio.late_exact_product) {
    return "LATE SKU";
  }
  if (portfolio.late_same_category) {
    return "LATE CAT";
  }
  return "NO HIT";
}

function firstPresetHouseholdId() {
  const portfolio = state.portfolios.find((item) => item.portfolio_id === state.presets[0].portfolio_id);
  return String(portfolio.household_id);
}

function renderPresets() {
  el.presetButtons.replaceChildren();
  let activeGroup = "";
  state.presets.forEach((preset) => {
    if (preset.group !== activeGroup) {
      activeGroup = preset.group;
      el.presetButtons.append(child("div", "preset-group-label", activeGroup));
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = `preset-button preset-${preset.tone}`;
    button.dataset.presetId = preset.portfolio_id;
    button.append(child("span", "preset-tag", preset.tag));
    button.append(child("strong", "", preset.label));
    button.append(child("small", "", preset.story));
    button.addEventListener("click", () => {
      const portfolio = state.portfolios.find((item) => item.portfolio_id === preset.portfolio_id);
      el.householdSelect.value = String(portfolio.household_id);
      renderTimeControl(preset.portfolio_id);
      loadSelectedPortfolio();
    });
    if (preset.portfolio_id === selectedPortfolio().portfolio_id) {
      button.classList.add("is-active");
    }
    el.presetButtons.append(button);
  });
}

function renderHeadlineMetrics() {
  const headline = state.metrics.headline;
  const late = state.metrics.late_evidence;
  text(el.metricRecall, score(headline.recall_at_10));
  text(el.metricNdcg, score(headline.ndcg_at_10));
  text(el.metricHit, pct(headline.positive_hit_at_10));
  text(el.metricLateExact, pct(late.late_exact_product_rate));
  text(el.metricLateCategory, pct(late.late_same_category_rate));
}

function renderMetricBars() {
  el.metricBars.replaceChildren();
  state.metrics.rows.forEach((row, index) => {
    const line = child("div", "bar-row");
    line.style.setProperty("--i", index);
    line.append(child("div", "bar-label", row.label));

    const track = child("div", "bar-track");
    const fill = child("div", "bar-fill");
    fill.style.width = `${Math.round(row.positive_event_hit_rate_at_10 * 100)}%`;
    track.append(fill);
    line.append(track);

    line.append(child("div", "bar-value", pct(row.positive_event_hit_rate_at_10)));
    el.metricBars.append(line);
  });
}

function loadSelectedPortfolio() {
  const portfolio = selectedPortfolio();
  updateTimeReadout(portfolio, Number(el.timeSlider.value), selectedHousehold().windows.length);
  fetch(`/api/recommendations?portfolio_id=${encodeURIComponent(portfolio.portfolio_id)}&coupon_slots=${COUPON_SLOTS}`)
    .then((response) => response.json())
    .then(renderPortfolio);
}

function renderPortfolio(data) {
  const event = data.event;
  const kpis = data.kpis;
  const preset = presetFor(event.event_id);
  text(el.eventTitle, `Household ${event.household_id}`);
  text(el.campaignType, event.campaign_type);
  text(el.modelName, event.model_name);
  text(el.couponStart, event.coupon_start_date);
  text(el.predictedPurchase, event.predicted_purchase_time.replace("T", " "));
  text(el.top10Hits, `${kpis.top10_observed_success} observed`);
  text(el.eligibleSlots, `${kpis.coupon_slots} selected`);
  text(el.avgScore, `Avg Top-10 score ${score(kpis.avg_top10_score)}`);
  renderStory(preset, kpis);
  renderPresets();
  renderRecommendations(data.recommendations);
  renderHistory(data.history, data.history_summary);
  renderLateEvidence(data.late_evidence);
}

function presetFor(eventId) {
  return state.presets.find((preset) => preset.portfolio_id === eventId);
}

function renderStory(preset, kpis) {
  const hitCount = kpis.top10_observed_success;
  document.body.dataset.eventMood = preset ? preset.tone : hitCount > 0 ? "hit" : "neutral";
  text(el.storyTag, preset ? preset.tag : hitCount > 0 ? `HIT ${hitCount}` : "HELD OUT");
  text(
    el.storyText,
    preset
      ? preset.story
      : "Held-out household portfolio for live browsing across coupon offers.",
  );
}

function renderRecommendations(rows) {
  el.recommendationRows.replaceChildren();
  rows.forEach((row, index) => {
    const tr = document.createElement("tr");
    tr.style.setProperty("--i", index);
    if (row.observed_success) {
      tr.classList.add("is-hit-row");
    }

    tr.append(cell(child("span", "rank", row.rank)));

    const product = document.createElement("div");
    product.append(child("span", "product-main", row.product_name));
    product.append(
      child(
        "span",
        "product-sub",
        `ID ${row.product_id} - ${row.brand} - ${row.department} / ${row.product_category}`,
      ),
    );
    product.append(child("span", "product-reason", row.reason));
    tr.append(cell(product));

    tr.append(cell(child("span", "", score(row.final_score))));
    tr.append(cell(flag(row.recommend_coupon, row.coupon_eligible, row.campaign_id)));
    tr.append(cell(observedFlag(row.observed_success)));

    el.recommendationRows.append(tr);
  });
}

function cell(content) {
  const td = document.createElement("td");
  td.append(content);
  return td;
}

function flag(recommended, eligible, campaignId) {
  if (recommended) {
    return child("span", "flag flag-on", eligible ? `coupon C${campaignId}` : "slot");
  }
  return child("span", "flag flag-off", eligible ? "eligible" : "no coupon");
}

function observedFlag(hit) {
  if (hit) {
    return child("span", "flag flag-hit", "hit");
  }
  return child("span", "flag flag-off", "not hit");
}

function renderHistory(rows, summary) {
  el.historyRows.replaceChildren();
  text(el.historyMeta, `Showing ${summary.shown} of ${summary.total_before_coupon}`);
  if (rows.length === 0) {
    const item = document.createElement("li");
    item.append(child("strong", "", "No prior purchase rows before coupon start"));
    item.append(child("span", "", `Cutoff ${summary.cutoff}`));
    el.historyRows.append(item);
    return;
  }
  rows.forEach((row, index) => {
    const item = document.createElement("li");
    item.style.setProperty("--i", index);
    item.append(child("strong", "", row.product_name));
    item.append(
      child(
        "span",
        "",
        `${row.purchase_time.replace("T", " ")} - ${row.department} / ${row.product_category} - Qty ${Number(row.quantity).toFixed(0)} - $${Number(row.sales_value).toFixed(2)}`,
      ),
    );
    el.historyRows.append(item);
  });
}

function renderLateEvidence(evidence) {
  el.lateEvidenceCards.replaceChildren();
  const global = state.metrics.late_evidence;
  text(
    el.lateEvidenceMeta,
    `${global.late_exact_product_windows}/${global.no_hit_windows} late SKU - ${global.late_same_category_windows}/${global.no_hit_windows} late category`,
  );

  const status = evidence.is_no_hit_window
    ? `No 5-day exact hit. Window ended ${evidence.response_window_end}.`
    : `This window already has a 5-day exact hit. Window ended ${evidence.response_window_end}.`;
  const statusCard = child("article", "late-evidence-card late-evidence-status");
  statusCard.append(child("span", "late-evidence-kicker", "Window status"));
  statusCard.append(child("strong", "", status));
  el.lateEvidenceCards.append(statusCard);

  el.lateEvidenceCards.append(
    lateCaseCard(
      "Later exact product",
      `${evidence.exact_product_case_count} Top-10 products`,
      evidence.exact_product_cases,
      "No later purchase of the exact recommended Top-10 products.",
    ),
  );
  el.lateEvidenceCards.append(
    lateCaseCard(
      "Later same category",
      `${evidence.same_category_case_count} recommended categories`,
      evidence.same_category_cases,
      "No later purchase in the recommended Top-10 categories.",
    ),
  );
}

function lateCaseCard(title, countLabel, rows, emptyText) {
  const card = child("article", "late-evidence-card");
  const header = child("div", "late-evidence-card-head");
  header.append(child("span", "late-evidence-kicker", title));
  header.append(child("strong", "", countLabel));
  card.append(header);

  if (rows.length === 0) {
    card.append(child("p", "late-evidence-empty", emptyText));
    return card;
  }

  const list = child("ol", "late-evidence-list");
  rows.forEach((row, index) => {
    const item = child("li");
    item.style.setProperty("--i", index);
    const tag = child("span", row.same_product ? "case-tag case-exact" : "case-tag case-category", row.kind);
    const titleLine = child("strong", "", row.purchased_product_name);
    const purchase = child(
      "span",
      "",
      `${row.purchase_time.replace("T", " ")} - ${Number(row.days_after_window).toFixed(1)} days after window - ${row.purchased_category} - $${Number(row.sales_value).toFixed(2)}`,
    );
    const matched = child(
      "small",
      "",
      `Matched Top-10 rank ${row.recommended_rank}: ${row.recommended_product_name} / ${row.recommended_category}`,
    );
    item.append(tag, titleLine, purchase, matched);
    list.append(item);
  });
  card.append(list);
  return card;
}

el.householdSelect.addEventListener("change", () => {
  renderTimeControl();
  loadSelectedPortfolio();
});
el.timeSlider.addEventListener("input", loadSelectedPortfolio);

fetch("/api/bootstrap")
  .then((response) => response.json())
  .then(setBootstrap);
