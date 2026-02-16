import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { Users, Activity, TrendingUp, Eye } from 'lucide-react';
import type { PostureSummaryData, PostureClass } from '../types';
import { POSTURE_COLORS } from '../types';

interface SessionSummaryProps {
  summary: PostureSummaryData;
}

export const SessionSummary: React.FC<SessionSummaryProps> = ({ summary }) => {
  const stats = [
    {
      label: 'Total Students',
      value: summary.total_students,
      icon: Users,
      color: 'text-blue-600',
      bg: 'bg-blue-100',
    },
    {
      label: 'Total Frames',
      value: summary.total_frames,
      icon: Activity,
      color: 'text-green-600',
      bg: 'bg-green-100',
    },
    {
      label: 'Most Common',
      value: summary.most_common_posture || 'N/A',
      icon: TrendingUp,
      color: 'text-purple-600',
      bg: 'bg-purple-100',
    },
    {
      label: 'Avg Confidence',
      value: `${(summary.average_confidence * 100).toFixed(1)}%`,
      icon: Eye,
      color: 'text-amber-600',
      bg: 'bg-amber-100',
    },
  ];

  const chartData = summary.posture_distribution.map((d) => ({
    name: d.posture,
    count: d.count,
    percentage: d.percentage,
    fill: POSTURE_COLORS[d.posture as PostureClass] || '#6b7280',
  }));

  return (
    <div className="space-y-6">
      {/* Stat Cards */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {stats.map((stat) => (
          <div key={stat.label} className="card flex items-center gap-4">
            <div className={`rounded-lg p-2.5 ${stat.bg}`}>
              <stat.icon className={`h-5 w-5 ${stat.color}`} />
            </div>
            <div>
              <p className="text-xs font-medium text-gray-500">{stat.label}</p>
              <p className="text-lg font-bold text-gray-900">{stat.value}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Distribution Chart */}
      <div className="card">
        <h3 className="mb-4 text-lg font-semibold text-gray-900">Posture Distribution</h3>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="name"
                angle={-35}
                textAnchor="end"
                interval={0}
                tick={{ fontSize: 11, fill: '#6b7280' }}
                height={80}
              />
              <YAxis tick={{ fontSize: 12, fill: '#6b7280' }} />
              <Tooltip
                contentStyle={{
                  borderRadius: '8px',
                  border: '1px solid #e5e7eb',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                }}
                formatter={(value: number, _name: string, props: { payload: { percentage: number } }) => [
                  `${value} (${props.payload.percentage}%)`,
                  'Count',
                ]}
              />
              <Bar dataKey="count" radius={[6, 6, 0, 0]} maxBarSize={60}>
                {chartData.map((entry, index) => (
                  <Cell key={index} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="py-8 text-center text-gray-500">No posture data available</p>
        )}
      </div>
    </div>
  );
};
