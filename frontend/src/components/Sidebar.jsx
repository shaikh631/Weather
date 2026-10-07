import { Activity, BarChart3, CloudSun, Database, Gauge, GitCompareArrows, History, Settings2, UploadCloud } from "lucide-react";

const links = [
  ["Dashboard", Gauge], ["Forecast", CloudSun], ["Model performance", BarChart3],
  ["Training history", History], ["Dataset", UploadCloud], ["Comparison", GitCompareArrows], ["Settings", Settings2],
];

export default function Sidebar({ active, onNavigate }) {
  return <aside className="sidebar">
    <div className="brand"><div className="brand-mark"><Activity size={20} /></div><div><strong>atmos</strong><small>WEATHER INTELLIGENCE</small></div></div>
    <div className="workspace-label">WORKSPACE</div>
    <nav>{links.map(([name, Icon]) => <button key={name} onClick={() => onNavigate(name)} className={`nav-item ${active === name ? "active" : ""}`}><Icon size={18} /><span>{name}</span>{name === "Forecast" && <span className="nav-pulse" />}</button>)}</nav>
    <div className="sidebar-bottom"><div className="sidebar-status"><span className="status-dot" /><div><b>Engine online</b><small>Local model active</small></div></div><small className="version-label">ATMOS ENGINE · v1.0.0</small></div>
  </aside>;
}
