const state = {
  events: [],
  presets: [],
  metrics: null,
};

const el = {
  eventSelect: document.querySelector("#eventSelect"),
  couponSlots: document.querySelector("#couponSlots"),
  couponSlotsValue: document.querySelector("#couponSlotsValue"),
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
  state.events = data.events;
  state.presets = data.presets;
  state.metrics = data.metrics;
  renderEventOptions();
  renderPresets();
  renderHeadlineMetrics();
  renderMetricBars();
  loadSelectedEvent();
}

function renderEventOptions() {
  el.eventSelect.replaceChildren();
  state.events.forEach((event) => {
    const option = document.createElement("option");
    const prefix = event.positive_in_top10 > 0 ? `HIT ${event.positive_in_top10}` : event.event_group === "history_showcase" ? "HISTORY" : "NO HIT";
    option.value = event.event_id;
    option.textContent = `${prefix} - HH ${event.household_id} - Campaign ${event.campaign_id} - ${event.top_product_category}`;
    el.eventSelect.append(option);
  });
}

function renderPresets() {
  el.presetButtons.replaceChildren();
  state.presets.forEach((preset) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `preset-button preset-${preset.tone}`;
    button.dataset.presetId = preset.event_id;
    button.append(child("span", "preset-tag", preset.tag));
    button.append(child("strong", "", preset.label));
    button.append(child("small", "", preset.story));
    button.addEventListener("click", () => {
      el.eventSelect.value = preset.event_id;
      el.couponSlots.value = preset.coupon_slots;
      loadSelectedEvent();
    });
    if (preset.event_id === el.eventSelect.value) {
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

function loadSelectedEvent() {
  const eventId = el.eventSelect.value;
  const couponSlots = el.couponSlots.value;
  text(el.couponSlotsValue, couponSlots);
  fetch(`/api/recommendations?event_id=${encodeURIComponent(eventId)}&coupon_slots=${couponSlots}`)
    .then((response) => response.json())
    .then(renderEvent);
}

function renderEvent(data) {
  const event = data.event;
  const kpis = data.kpis;
  const preset = presetFor(event.event_id);
  text(el.eventTitle, `Household ${event.household_id} - Campaign ${event.campaign_id}`);
  text(el.campaignType, event.campaign_type);
  text(el.modelName, event.model_name);
  text(el.couponStart, event.coupon_start_date);
  text(el.predictedPurchase, event.predicted_purchase_time.replace("T", " "));
  text(el.top10Hits, `${kpis.top10_observed_success} observed`);
  text(el.eligibleSlots, `${kpis.coupon_eligible_in_slots} of ${kpis.coupon_slots}`);
  text(el.avgScore, `Avg Top-10 score ${score(kpis.avg_top10_score)}`);
  renderStory(preset, kpis);
  renderPresets();
  renderRecommendations(data.recommendations);
  renderHistory(data.history);
}

function presetFor(eventId) {
  return state.presets.find((preset) => preset.event_id === eventId);
}

function renderStory(preset, kpis) {
  const hitCount = kpis.top10_observed_success;
  document.body.dataset.eventMood = preset ? preset.tone : hitCount > 0 ? "hit" : "neutral";
  text(el.storyTag, preset ? preset.tag : hitCount > 0 ? `HIT ${hitCount}` : "HELD OUT");
  text(
    el.storyText,
    preset
      ? preset.story
      : "Held-out test event for live browsing across household-campaign recommendations.",
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
    tr.append(cell(flag(row.recommend_coupon, row.coupon_eligible)));
    tr.append(cell(observedFlag(row.observed_success)));

    el.recommendationRows.append(tr);
  });
}

function cell(content) {
  const td = document.createElement("td");
  td.append(content);
  return td;
}

function flag(recommended, eligible) {
  if (recommended) {
    return child("span", "flag flag-on", eligible ? "coupon" : "slot");
  }
  return child("span", "flag flag-off", eligible ? "eligible" : "no coupon");
}

function observedFlag(hit) {
  if (hit) {
    return child("span", "flag flag-hit", "hit");
  }
  return child("span", "flag flag-off", "not hit");
}

function renderHistory(rows) {
  el.historyRows.replaceChildren();
  if (rows.length === 0) {
    const item = document.createElement("li");
    item.append(child("strong", "", "No matched history rows for this final test event"));
    item.append(child("span", "", "Use the presentation presets for controlled hit and miss examples."));
    el.historyRows.append(item);
    return;
  }
  rows.forEach((row, index) => {
    const item = document.createElement("li");
    item.style.setProperty("--i", index);
    item.append(child("strong", "", row.product_name));
    item.append(child("span", "", `${row.purchase_time.replace("T", " ")} - ${row.department} / ${row.product_category}`));
    el.historyRows.append(item);
  });
}

el.eventSelect.addEventListener("change", loadSelectedEvent);
el.couponSlots.addEventListener("input", loadSelectedEvent);

fetch("/api/bootstrap")
  .then((response) => response.json())
  .then(setBootstrap);
