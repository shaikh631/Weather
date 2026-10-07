import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";

const axis = { stroke: "#8a94a6", fontSize: 11, tickLine: false, axisLine: false };
const tooltipStyle = { border: "1px solid #e5eaf1", borderRadius: 12, boxShadow: "0 10px 30px #17243b12" };

export function AccuracyChart({ data = [] }) {
  const rows = data.map((item) => ({ ...item, label: item.version }));
  return <ResponsiveContainer width="100%" height="100%"><AreaChart data={rows} margin={{ top: 12, right: 8, bottom: 0, left: -20 }}>
    <defs><linearGradient id="accuracyFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#2563eb" stopOpacity={0.18} /><stop offset="100%" stopColor="#2563eb" stopOpacity={0} /></linearGradient></defs>
    <CartesianGrid stroke="#edf0f5" vertical={false} /><XAxis dataKey="label" tick={axis} /><YAxis tick={axis} domain={["auto", "auto"]} />
    <Tooltip contentStyle={tooltipStyle} formatter={(value) => [Number(value).toFixed(3), "R² score"]} />
    <Area type="monotone" dataKey="r2_score" stroke="#2563eb" strokeWidth={2.5} fill="url(#accuracyFill)" />
  </AreaChart></ResponsiveContainer>;
}

export function MetricTrendChart({ data = [], metric }) {
  return <ResponsiveContainer width="100%" height="100%"><LineChart data={data} margin={{ top: 12, right: 8, bottom: 0, left: -20 }}>
    <CartesianGrid stroke="#edf0f5" vertical={false} /><XAxis dataKey="version" tick={axis} /><YAxis tick={axis} />
    <Tooltip contentStyle={tooltipStyle} /><Line type="monotone" dataKey={metric} stroke={metric === "rmse" ? "#f59e0b" : "#10b981"} strokeWidth={2.5} dot={{ r: 4 }} />
  </LineChart></ResponsiveContainer>;
}

export function ComparisonChart({ data = [] }) {
  return <ResponsiveContainer width="100%" height="100%"><BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -20 }}>
    <CartesianGrid stroke="#edf0f5" vertical={false} /><XAxis dataKey="model" tick={axis} /><YAxis tick={axis} /><Tooltip contentStyle={tooltipStyle} />
    <Bar dataKey="r2" name="R² score" fill="#4f76ed" radius={[6, 6, 0, 0]} />
  </BarChart></ResponsiveContainer>;
}

export function ActualChart({ data = [], target = "temperature" }) {
  const rows = data.filter((item) => item.target === target).map((item, index) => ({ ...item, index: index + 1 }));
  return <ResponsiveContainer width="100%" height="100%"><LineChart data={rows} margin={{ top: 12, right: 8, bottom: 0, left: -20 }}>
    <CartesianGrid stroke="#edf0f5" vertical={false} /><XAxis dataKey="index" tick={axis} /><YAxis tick={axis} /><Tooltip contentStyle={tooltipStyle} />
    <Line type="monotone" dataKey="actual" name={`Actual ${target}`} stroke="#25334c" strokeWidth={2} dot={false} />
    <Line type="monotone" dataKey="predicted" name={`Predicted ${target}`} stroke="#4f76ed" strokeWidth={2} dot={false} />
  </LineChart></ResponsiveContainer>;
}

export function ResidualChart({ data = [], target = "temperature" }) {
  const rows = data.filter((item) => item.target === target);
  return <ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 12, right: 15, bottom: 4, left: -12 }}>
    <CartesianGrid stroke="#edf0f5" /><XAxis type="number" dataKey="actual" name="Actual" tick={axis} /><YAxis type="number" dataKey="residual" name="Residual" tick={axis} />
    <ReferenceLine y={0} stroke="#8e9bb0" strokeDasharray="4 4" /><Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={tooltipStyle} />
    <Scatter data={rows} fill="#e07870" />
  </ScatterChart></ResponsiveContainer>;
}

export function FeatureChart({ data = [] }) {
  return <ResponsiveContainer width="100%" height="100%"><BarChart data={data.slice(0, 7)} layout="vertical" margin={{ top: 0, right: 20, bottom: 0, left: 16 }}>
    <CartesianGrid stroke="#edf0f5" horizontal={false} /><XAxis type="number" tick={axis} /><YAxis type="category" dataKey="feature" tick={axis} width={130} /><Tooltip contentStyle={tooltipStyle} />
    <Bar dataKey="importance" fill="#31a88d" radius={[0, 5, 5, 0]} />
  </BarChart></ResponsiveContainer>;
}
