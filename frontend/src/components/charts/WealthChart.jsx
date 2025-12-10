import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

const WealthChart = ({ data }) => {
  const { stats } = data;
  const wealthData = stats.wealth_index;

  return (
    <div className="h-96 w-full bg-white p-6 rounded-sm border border-beige shadow-sm">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={wealthData}
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#C5A059" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#C5A059" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#F0EBE0" vertical={false} />
          <XAxis
            dataKey="date"
            stroke="#8C735A"
            tick={{ fill: '#2C3E50', fontSize: 12, fontFamily: 'Inter' }}
            tickFormatter={(str) => str.substring(0, 4)}
            minTickGap={50}
          />
          <YAxis
            stroke="#8C735A"
            tick={{ fill: '#2C3E50', fontSize: 12, fontFamily: 'Inter' }}
            tickFormatter={(value) => `R${value.toLocaleString()}`}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#F9F7F2', borderColor: '#C5A059', fontFamily: 'Inter' }}
            itemStyle={{ color: '#1A2433' }}
            formatter={(value) => [`R${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, 'Value']}
            labelFormatter={(label) => new Date(label).toLocaleDateString(undefined, { year: 'numeric', month: 'long' })}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke="#C5A059"
            strokeWidth={2}
            fillOpacity={1}
            fill="url(#colorValue)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default WealthChart;
