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
    Number(a.positive_in_top10) - Number(b.positive_in_top10)
  );
}

function renderHouseholdOptions() {
  el.householdSelect.replaceChildren();
  state.households.forEach((household) => {
    const option = document.createElement("option");
    const best = household.best;
    const prefix = best.positive_in_top10 > 0 ? `HIT ${best.positive_in_top10}` : "NO HIT";
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
  text(el.timeWindowMeta, `${index + 1} of ${total} windows - ${portfolio.positive_in_top10} observed hits`);
}

function firstPresetHouseholdId() {
  const portfolio = state.portfolios.find((item) => item.portfolio_id === state.presets[0].portfolio_id);
  return String(portfolio.household_id);
}

function renderPresets() {
  el.presetButtons.replaceChildren();
  state.presets.forEach((preset) => {
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
  text(el.metricRecall, score(headline.recall_at_10));
  text(el.metricNdcg, score(headline.ndcg_at_10));
  text(el.metricHit, pct(headline.positive_hit_at_10));
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

el.householdSelect.addEventListener("change", () => {
  renderTimeControl();
  loadSelectedPortfolio();
});
el.timeSlider.addEventListener("input", loadSelectedPortfolio);

fetch("/api/bootstrap")
  .then((response) => response.json())
  .then(setBootstrap);
