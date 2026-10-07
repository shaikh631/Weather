import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowDownRight, ArrowUpRight, CloudDrizzle, CloudRain, CloudSun, Droplets, MapPin,
  Database, GitCompareArrows, Menu, Moon, RefreshCw, Search, Sun, UploadCloud, Wind,
} from "lucide-react";
import Sidebar from "./components/Sidebar.jsx";
import StatCard from "./components/StatCard.jsx";
import { AccuracyChart, ActualChart, ComparisonChart, FeatureChart, MetricTrendChart, ResidualChart } from "./components/Charts.jsx";
import { api } from "./services/api.js";

const initialLocation = { city: "Mumbai", latitude: "19.0760", longitude: "72.8777" };
const navPages = {
  Dashboard: "Your localized climate intelligence, at a glance.",
  Forecast: "Generate a location-aware forecast from your current conditions.",
  "Model performance": "Track model quality and inspect the latest evaluation.",
  "Training history": "Every retraining run, versioned and preserved.",
  Dataset: "Validate weather data and retrain your forecasting engine.",
  Comparison: "Compare candidate regressors on the same held-out observations.",
  Settings: "Configure the workspace and inspect service status.",
};
const number = (value, places = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(places) : "—";

function Panel({ title, subtitle, action, children, className = "" }) {
  return <section className={`panel ${className}`}><div className="panel-head"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</div>{children}</section>;
}

function EmptyChart({ children = "Train a model to see performance data here." }) {
  return <div className="empty-chart"><span>◌</span>{children}</div>;
}

function ChartSkeleton() {
  return <div className="chart-skeleton" aria-label="Loading chart"><i /><i /><i /></div>;
}

export default function App() {
  const [active, setActive] = useState("Dashboard");
  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState([]);
  const [comparison, setComparison] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [forecast, setForecast] = useState(null);
  const [location, setLocation] = useState(initialLocation);
  const [conditions, setConditions] = useState({ temperature: "27", humidity: "70", rainfall: "0", wind_speed: "12", pressure: "1013" });
  const [dataset, setDataset] = useState(null);
  const [health, setHealth] = useState(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const [busy, setBusy] = useState("");
  const [toast, setToast] = useState("");
  const [dark, setDark] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [targetChart, setTargetChart] = useState("temperature");

  const refresh = useCallback(async () => {
    const [latestResult, historyResult, comparisonResult, predictionResult, healthResult] = await Promise.all([
      api.latest(), api.history(), api.comparison(), api.weatherHistory(), api.health(),
    ]);
    setLatest(latestResult.model);
    setHistory(historyResult.iterations);
    setComparison(comparisonResult.models);
    setPredictions(predictionResult.predictions);
    setHealth(healthResult);
  }, []);

  useEffect(() => {
    refresh().catch((error) => setToast(error.message)).finally(() => setInitialLoading(false));
  }, [refresh]);
  useEffect(() => {
    if (!toast) return undefined;
    const timer = setTimeout(() => setToast(""), 4200);
    return () => clearTimeout(timer);
  }, [toast]);

  const previous = history.length > 1 ? history[history.length - 2] : null;
  const improvement = previous && latest ? ((latest.r2_score - previous.r2_score) / Math.max(Math.abs(previous.r2_score), 0.01)) * 100 : null;
  const chartHistory = useMemo(() => history.map((row) => ({ ...row, r2_score: row.r2_score * 100, label: row.version })), [history]);

  async function runForecast(event) {
    event.preventDefault();
    setBusy("forecast");
    try {
      const result = await api.predict({
        city: location.city, latitude: Number(location.latitude), longitude: Number(location.longitude),
        ...Object.fromEntries(Object.entries(conditions).map(([key, value]) => [key, Number(value)])),
      });
      setForecast(result);
      setToast(`Forecast generated for ${result.city}`);
      await refresh();
    } catch (error) {
      setToast(error.message);
    } finally {
      setBusy("");
    }
  }

  async function retrain() {
    setBusy("train");
    try {
      const result = await api.train();
      setDataset(null);
      setToast(`${result.model.version} trained successfully · ${result.model.model_name}`);
      await refresh();
    } catch (error) {
      setToast(error.message);
    } finally {
      setBusy("");
    }
  }

  async function uploadFile(file) {
    if (!file) return;
    setBusy("upload");
    try {
      const result = await api.upload(file);
      setDataset(result.stats);
      setToast(`Validated ${result.stats.records.toLocaleString()} weather records`);
    } catch (error) {
      setToast(error.message);
    } finally {
      setBusy("");
    }
  }

  const filteredHistory = history.filter((item) => `${item.version} ${item.model_name} ${item.training_date}`.toLowerCase().includes(query.toLowerCase()));
  const pageTitle = active;
  const Header = <header className="topbar">
    <button className="mobile-menu icon-button" onClick={() => setMobileOpen(!mobileOpen)} aria-label="Toggle navigation"><Menu size={20} /></button>
    <div className="breadcrumbs"><span>Workspace</span><b>/</b><strong>{active}</strong></div>
    <div className="top-actions">
      <span className="header-live"><i /> Live engine</span>
      <button className="icon-button theme-button" title="Toggle theme" onClick={() => setDark(!dark)}>{dark ? <Sun size={18} /> : <Moon size={18} />}</button>
      <div className="avatar">AS</div>
    </div>
  </header>;

  return <div className={`app-shell ${dark ? "dark" : ""}`}>
    <div className={`sidebar-wrap ${mobileOpen ? "open" : ""}`}><Sidebar active={active} onNavigate={(value) => { setActive(value); setMobileOpen(false); }} /></div>
    <main className="main-area">
      {Header}
      <div className="page-content">
        <div className="page-heading"><div><div className="eyebrow"><span className="eyebrow-dot" /> LOCALIZED WEATHER INTELLIGENCE</div><h1>{pageTitle}</h1><p>{navPages[active]}</p></div>
          {active !== "Settings" && <button className="button primary" disabled={Boolean(busy)} onClick={retrain}><RefreshCw size={16} className={busy === "train" ? "spinning" : ""} />{busy === "train" ? "Training…" : "Retrain model"}</button>}
        </div>
        {active === "Dashboard" && <>
          <section className="welcome-banner"><div className="welcome-copy"><div className="banner-kicker"><span className="weather-spark" /> YOUR LOCAL OUTLOOK</div><h2>Weather, made <em>predictable.</em></h2><p>Machine learning forecasts tuned to your location, continuously improving with every training run.</p><button className="banner-link" onClick={() => setActive("Forecast")}>Create a forecast <ArrowUpRight size={16} /></button></div><div className="banner-art"><div className="sun-orb" /><div className="cloud cloud-one" /><div className="cloud cloud-two" /><div className="banner-location"><MapPin size={15} /> Mumbai, IN <span>19.07° N</span></div></div></section>
          <div className="stats-grid">
            <StatCard label="Temperature" value={forecast ? number(forecast.predictions.temperature, 1) : "28.6"} unit="°C" note={forecast ? `Forecast · ${forecast.city}` : "Demo reference · Mumbai"} icon={Sun} tone="amber" />
            <StatCard label="Predicted temperature" value={forecast ? number(forecast.predictions.temperature, 1) : "—"} unit={forecast ? "°C" : ""} note={forecast ? `${forecast.city} · next hour` : "Generate a location forecast"} icon={CloudSun} tone="blue" />
            <StatCard label="Humidity" value={forecast ? number(forecast.predictions.humidity, 0) : "72"} unit="%" note={forecast ? `Forecast · ${forecast.city}` : "Illustrative reference"} icon={Droplets} tone="blue" />
            <StatCard label="Rainfall" value={forecast ? number(forecast.predictions.rainfall, 1) : "1.2"} unit="mm" note={forecast ? "Next-hour estimate" : "Illustrative reference"} icon={CloudRain} tone="violet" />
            <StatCard label="Wind speed" value={forecast ? number(forecast.predictions.wind_speed, 1) : "13.4"} unit="km/h" note={forecast ? "Next-hour estimate" : "Illustrative reference"} icon={Wind} tone="teal" />
            <StatCard label="Model accuracy" value={latest ? number(latest.r2_score * 100, 1) : "—"} unit="%" note={latest ? `${latest.model_name} · ${latest.version}` : "Train to calculate"} icon={ActivityIcon} tone="blue" />
          </div>
          <div className="dashboard-grid">
            <Panel title="Model accuracy" subtitle="R² score across training iterations" action={<span className="chart-legend"><i /> R² SCORE</span>} className="accuracy-panel">
              <div className="chart-large">{initialLoading ? <ChartSkeleton /> : chartHistory.length ? <AccuracyChart data={chartHistory} /> : <EmptyChart />}</div>
              <div className="chart-summary"><span><small>CURRENT ACCURACY</small><b>{latest ? `${number(latest.r2_score * 100, 1)}%` : "—"}</b></span><span><small>PREVIOUS ACCURACY</small><b>{previous ? `${number(previous.r2_score * 100, 1)}%` : "—"}</b></span><span><small>IMPROVEMENT</small><b className={improvement >= 0 ? "positive" : "negative"}>{improvement == null ? "—" : `${improvement >= 0 ? "+" : ""}${number(improvement, 1)}%`} {improvement >= 0 ? <ArrowUpRight size={15} /> : <ArrowDownRight size={15} />}</b></span><span><small>TRAINING RUNS</small><b>{history.length}</b></span></div>
            </Panel>
            <Panel title="Current model" subtitle="Production candidate" action={<span className="model-badge">{latest?.version || "READY"}</span>} className="model-panel">
              {latest ? <><div className="model-name">{latest.model_name}</div><div className="model-subtitle">Multi-output regression · 4 targets</div><div className="model-metrics"><div><small>R² SCORE</small><b>{number(latest.r2_score, 3)}</b></div><div><small>RMSE</small><b>{number(latest.rmse)}</b></div><div><small>MAE</small><b>{number(latest.mae)}</b></div></div><div className="model-foot"><span><Database size={15} /> {latest.dataset_size?.toLocaleString()} observations</span><span>{latest.features} features</span></div><button className="text-button" onClick={() => setActive("Model performance")}>View model details <ArrowUpRight size={14} /></button></> : initialLoading ? <ChartSkeleton /> : <EmptyChart>Model training on startup… refresh shortly.</EmptyChart>}
            </Panel>
          </div>
          <div className="dashboard-grid lower-grid">
            <Panel title="Actual vs predicted" subtitle={`${targetChart} · held-out observations`} action={<div className="target-tabs">{["temperature", "humidity", "rainfall", "wind_speed"].map((target) => <button key={target} className={targetChart === target ? "selected" : ""} onClick={() => setTargetChart(target)}>{target === "wind_speed" ? "Wind" : target}</button>)}</div>} className="actual-panel"><div className="chart-medium">{initialLoading ? <ChartSkeleton /> : latest?.actual_vs_predicted?.length ? <ActualChart data={latest.actual_vs_predicted} target={targetChart} /> : <EmptyChart />}</div></Panel>
            <Panel title="Feature importance" subtitle={`What drives ${latest?.model_name || "the model"} predictions`} className="feature-panel"><div className="chart-medium">{initialLoading ? <ChartSkeleton /> : latest?.feature_importance?.length ? <FeatureChart data={latest.feature_importance} /> : <EmptyChart />}</div></Panel>
          </div>
        </>}

        {active === "Forecast" && <div className="forecast-layout">
          <Panel title="Location forecast" subtitle="Set coordinates and current observations to forecast the next hour" className="form-panel">
            <form onSubmit={runForecast}>
              <label className="field-label">Location</label><div className="input-with-icon"><Search size={17} /><input value={location.city} onChange={(e) => setLocation({ ...location, city: e.target.value })} required maxLength="100" placeholder="Search city" /></div>
              <div className="form-row"><label>Latitude<input type="number" step="any" min="-90" max="90" value={location.latitude} onChange={(e) => setLocation({ ...location, latitude: e.target.value })} required /></label><label>Longitude<input type="number" step="any" min="-180" max="180" value={location.longitude} onChange={(e) => setLocation({ ...location, longitude: e.target.value })} required /></label></div>
              <div className="form-divider"><span>Current conditions</span><i /></div>
              <div className="form-row"><label>Temperature (°C)<input type="number" step="any" value={conditions.temperature} onChange={(e) => setConditions({ ...conditions, temperature: e.target.value })} required /></label><label>Humidity (%)<input type="number" min="0" max="100" value={conditions.humidity} onChange={(e) => setConditions({ ...conditions, humidity: e.target.value })} required /></label></div>
              <div className="form-row"><label>Rainfall (mm)<input type="number" min="0" step="any" value={conditions.rainfall} onChange={(e) => setConditions({ ...conditions, rainfall: e.target.value })} required /></label><label>Wind (km/h)<input type="number" min="0" step="any" value={conditions.wind_speed} onChange={(e) => setConditions({ ...conditions, wind_speed: e.target.value })} required /></label></div>
              <label className="field-label">Pressure (hPa)<input type="number" min="800" max="1200" value={conditions.pressure} onChange={(e) => setConditions({ ...conditions, pressure: e.target.value })} required /></label>
              <button className="button primary full-button" disabled={Boolean(busy)}>{busy === "forecast" ? "Generating forecast…" : <><CloudSun size={17} /> Generate forecast</>}</button>
            </form>
          </Panel>
          <div className="forecast-results">
            {forecast ? <Panel title="Forecast result" subtitle={`${forecast.city} · ${new Date(forecast.predicted_at).toLocaleString()}`} className="result-panel">
              <div className="result-hero"><div className="forecast-sun"><Sun size={36} /></div><div><strong>{number(forecast.predictions.temperature, 1)}°</strong><span>Next-hour temperature</span></div><div className="forecast-location"><MapPin size={15} /> {forecast.latitude.toFixed(2)}, {forecast.longitude.toFixed(2)}</div></div>
              <div className="result-grid">{[["Humidity", "humidity", "%", Droplets], ["Rainfall", "rainfall", "mm", CloudDrizzle], ["Wind speed", "wind_speed", "km/h", Wind]].map(([label, key, unit, Icon]) => <div key={key} className="result-metric"><Icon size={18} /><span>{label}</span><b>{number(forecast.predictions[key], 1)} <small>{unit}</small></b><small>± {number(forecast.uncertainty[key], 1)} {unit} uncertainty</small></div>)}</div>
              <div className="result-model">Generated by <b>{forecast.model_name}</b><span>{forecast.version}</span></div>
            </Panel> : <div className="forecast-empty"><div className="empty-weather"><CloudSun size={34} /></div><h3>Your next-hour forecast</h3><p>Enter a location and current conditions to generate a model-powered localized prediction.</p></div>}
            <Panel title="Recent predictions" subtitle="Forecast requests from this workspace" className="recent-panel">
              {predictions.length ? predictions.slice(0, 5).map((item, index) => <div className="recent-row" key={`${item.predicted_at}-${index}`}><div className="recent-icon"><CloudSun size={17} /></div><div><b>{item.city}</b><small>{new Date(item.predicted_at).toLocaleString()}</small></div><strong>{number(item.predictions.temperature, 1)}°C</strong></div>) : <EmptyChart>No forecasts generated yet.</EmptyChart>}
            </Panel>
          </div>
        </div>}

        {active === "Model performance" && <><div className="stats-grid performance-stats"><StatCard label="R² score" value={latest ? number(latest.r2_score, 3) : "—"} note="Held-out test set" icon={ActivityIcon} /><StatCard label="RMSE" value={latest ? number(latest.rmse) : "—"} note="Root mean squared error" icon={GaugeIcon} tone="amber" /><StatCard label="MAE" value={latest ? number(latest.mae) : "—"} note="Mean absolute error" icon={CloudDrizzle} tone="teal" /><StatCard label="Test MAPE" value={latest ? `${number(Object.values(latest.mape || {}).reduce((a, b) => a + b, 0) / Math.max(Object.keys(latest.mape || {}).length, 1), 1)}%` : "—"} note="Mean target-level MAPE" icon={ActivityIcon} tone="violet" /></div>
          <div className="dashboard-grid"><Panel title="R² score by iteration" subtitle="Higher is better" className="accuracy-panel"><div className="chart-large">{chartHistory.length ? <AccuracyChart data={chartHistory} /> : <EmptyChart />}</div></Panel><Panel title="RMSE by iteration" subtitle="Lower is better" className="model-panel"><div className="chart-large">{history.length ? <MetricTrendChart data={history} metric="rmse" /> : <EmptyChart />}</div></Panel></div>
          <div className="dashboard-grid lower-grid"><Panel title="Actual vs predicted" subtitle={`${targetChart} · held-out observations`} action={<div className="target-tabs">{["temperature", "humidity", "rainfall", "wind_speed"].map((target) => <button key={target} className={targetChart === target ? "selected" : ""} onClick={() => setTargetChart(target)}>{target === "wind_speed" ? "Wind" : target}</button>)}</div>}><div className="chart-medium">{latest?.actual_vs_predicted?.length ? <ActualChart data={latest.actual_vs_predicted} target={targetChart} /> : <EmptyChart />}</div></Panel><Panel title="Feature importance" subtitle="Top model drivers"><div className="chart-medium">{latest?.feature_importance?.length ? <FeatureChart data={latest.feature_importance} /> : <EmptyChart />}</div></Panel></div>
          <div className="dashboard-grid lower-grid"><Panel title="Residual distribution" subtitle={`${targetChart} · error vs actual`}><div className="chart-medium">{latest?.residuals?.length ? <ResidualChart data={latest.residuals} target={targetChart} /> : <EmptyChart />}</div></Panel><Panel title="MAE trend" subtitle="Error tracked over retraining runs"><div className="chart-medium">{history.length ? <MetricTrendChart data={history} metric="mae" /> : <EmptyChart />}</div></Panel></div></>}

        {active === "Training history" && <Panel title="Training runs" subtitle={`${history.length} immutable model version${history.length === 1 ? "" : "s"} recorded`} action={<div className="table-search"><Search size={15} /><input placeholder="Search runs…" value={query} onChange={(e) => setQuery(e.target.value)} /></div>} className="table-panel"><div className="table-scroll"><table><thead><tr><th>VERSION</th><th>TRAINED</th><th>DATASET</th><th>MODEL</th><th>R² SCORE</th><th>RMSE</th><th>MAE</th></tr></thead><tbody>{filteredHistory.map((row) => <tr key={row.version}><td><span className="version-pill">{row.version}</span></td><td>{new Date(row.training_date).toLocaleString()}</td><td>{row.dataset_size?.toLocaleString()} rows</td><td>{row.model_name}</td><td><b>{number(row.r2_score, 3)}</b></td><td>{number(row.rmse)}</td><td>{number(row.mae)}</td></tr>)}</tbody></table>{!filteredHistory.length && <EmptyChart>No training iterations match.</EmptyChart>}</div></Panel>}

        {active === "Dataset" && <div className="dataset-layout"><Panel title="Upload weather data" subtitle="CSV only · up to 10 MB · required columns: date, latitude, longitude, temperature, humidity, rainfall, wind_speed, pressure" className="upload-panel"><label className="dropzone"><input type="file" accept=".csv,text/csv" onChange={(e) => uploadFile(e.target.files?.[0])} /><div className="upload-icon"><UploadCloud size={25} /></div><b>{busy === "upload" ? "Validating your dataset…" : "Drop your CSV here or browse"}</b><span>CSV datasets are validated and previewed before training</span></label>{dataset && <><div className="dataset-overview"><div><small>VALID RECORDS</small><b>{dataset.records.toLocaleString()}</b></div><div><small>MISSING VALUES</small><b>{dataset.missing_values}</b></div><div><small>DUPLICATES REMOVED</small><b>{dataset.duplicate_records}</b></div><div><small>DETECTED OUTLIERS</small><b>{Object.values(dataset.outliers).reduce((a, b) => a + b, 0)}</b></div></div><div className="table-scroll preview-table"><table><thead><tr>{Object.keys(dataset.preview[0] || {}).map((key) => <th key={key}>{key.toUpperCase()}</th>)}</tr></thead><tbody>{dataset.preview.map((row, i) => <tr key={i}>{Object.values(row).map((value, j) => <td key={j}>{String(value ?? "—")}</td>)}</tr>)}</tbody></table></div><button className="button primary" disabled={Boolean(busy)} onClick={retrain}><RefreshCw size={15} /> Train with uploaded dataset</button></>}</Panel><Panel title="Dataset checklist" subtitle="Quality gates applied on ingestion"><div className="checklist">{["Required schema check", "Timestamp & coordinate validation", "Duplicate timestamp checks", "Physical-range outlier detection", "Missing value imputation", "Chronological hold-out split", "Temporal cross-validation"].map((item, i) => <div key={item}><span className="check-number">0{i + 1}</span><span>{item}</span><i>Applied</i></div>)}</div><div className="dataset-note"><b>Preventing target leakage</b><p>Features at time t forecast the next observation. The hold-out set is the latest 20% of records, and preprocessing is fitted inside the model pipeline.</p></div></Panel></div>}

        {active === "Comparison" && <><Panel title="Candidate model comparison" subtitle="Same temporal hold-out set · multi-output metrics averaged across four targets"><div className="comparison-layout"><div className="comparison-chart">{comparison.length ? <ComparisonChart data={comparison} /> : <EmptyChart />}</div><div className="comparison-table table-scroll"><table><thead><tr><th>MODEL</th><th>MAE</th><th>RMSE</th><th>R²</th><th>CV R²</th><th>FIT TIME</th></tr></thead><tbody>{comparison.map((row) => <tr key={row.model} className={latest?.model_name === row.model ? "best-row" : ""}><td>{row.model}{latest?.model_name === row.model && <span className="best-chip">SELECTED</span>}</td><td>{number(row.mae)}</td><td>{number(row.rmse)}</td><td>{number(row.r2, 3)}</td><td>{number(row.cv_r2, 3)}</td><td>{number(row.training_time, 2)}s</td></tr>)}</tbody></table></div></div></Panel><div className="comparison-note"><GitCompareArrows size={19} /><p><b>Model selection objective</b><br />The highest held-out R² candidate is selected, with cross-validation scores shown for stability context. Hyperparameter search is bounded and applied to the selected Random Forest family where applicable.</p></div></>}

        {active === "Settings" && <div className="settings-grid"><Panel title="Workspace status" subtitle="Runtime and persistence diagnostics"><div className="settings-row"><div><b>Forecast API</b><small>FastAPI service health</small></div><span className="health-chip"><i /> {health?.status || "Checking"}</span></div><div className="settings-row"><div><b>Model persistence</b><small>Joblib versioned artifact storage</small></div><span className="settings-value">{latest?.version || "Not trained"}</span></div><div className="settings-row"><div><b>Training history store</b><small>MongoDB Atlas or local in-memory adapter</small></div><span className="settings-value">{health?.database || "—"}</span></div><div className="settings-row"><div><b>Active model</b><small>Selected by held-out performance</small></div><span className="settings-value">{latest?.model_name || "—"}</span></div></Panel><Panel title="Forecast configuration" subtitle="Current local defaults"><div className="settings-row"><div><b>Default location</b><small>Latitude and longitude used in the forecast form</small></div><span className="settings-value">Mumbai · 19.076, 72.878</span></div><div className="settings-row"><div><b>Prediction horizon</b><small>Next observation in the hourly weather series</small></div><span className="settings-value">1 hour</span></div><div className="settings-row"><div><b>Targets</b><small>Multi-output regression model</small></div><span className="settings-value">4 weather variables</span></div><button className="button secondary settings-refresh" onClick={() => refresh().catch((e) => setToast(e.message))}><RefreshCw size={15} /> Refresh service status</button></Panel><div className="settings-tip"><b>Environment-based configuration</b><p>Backend secrets and service URLs are read from environment variables. Never add your MongoDB URI to frontend configuration.</p></div></div>}
      </div>
      <footer className="footer"><span>© 2025 Atmos Weather Intelligence</span><span>Built for better local forecasts <CloudSun size={14} /></span></footer>
    </main>
    {toast && <div className={`toast ${toast.toLowerCase().includes("fail") || toast.toLowerCase().includes("error") ? "error" : ""}`} role="status"><span />{toast}<button onClick={() => setToast("")}>×</button></div>}
    {mobileOpen && <button className="mobile-backdrop" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
  </div>;
}

function ActivityIcon(props) { return <CloudSun {...props} />; }
function GaugeIcon(props) { return <Wind {...props} />; }
