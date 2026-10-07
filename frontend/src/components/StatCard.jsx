export default function StatCard({ label, value, unit, note, icon: Icon, tone = "blue" }) {
  return <article className="stat-card">
    <div className="stat-top"><span>{label}</span><span className={`stat-icon ${tone}`}><Icon size={17} /></span></div>
    <div className="stat-value">{value}<small>{unit}</small></div>
    <div className="stat-note">{note}</div>
  </article>;
}
