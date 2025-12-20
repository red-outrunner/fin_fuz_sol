import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

const WealthChart = ({ data }) => {
  if (!data || !data.stats || !data.stats.wealth_index) {
    return <div className="p-6 text-center text-slate-500 italic">No wealth data available</div>;
  }

  const { stats, moving_averages } = data;
  const wealthData = stats.wealth_index;

  // We need to fuse MA data with wealth data for the chart, OR just pass prices if we want to show price chart. 
  // Wait, Wealth Index is synthesized. MA is on Price.
  // Ideally we should show the Price chart for MAs, or calculate MA on the Wealth Index.
  // Calculating MA on Wealth Index is better for "Growth of 10k".
  // Let's create a derivative dataset on the fly or assuming the backend could send it.
  // Backend sent MAs on *Price*.
  // Let's just overlay MAs on a *Price* chart? 
  // User asked for "Moving Averages". Usually on the main price chart.
  // But our main chart IS the Wealth Index.
  // Let's just calculate simple MAs on the Wealth Index data client-side for visualization?
  // Actually, let's use the 'prices' and 'moving_averages' data from backend to show a secondary "Price History" chart?
  // OR, better, let's just stick to the plan: "Render lines on the main wealth chart".
  // Use `recharts` to process? No.
  // Let's just accept we need to calculate SMA on the wealth index values.

  const [showSMA12, setShowSMA12] = React.useState(false);
  const [showSMA60, setShowSMA60] = React.useState(false);

  // Helper to calculate SMA on the wealth index array
  const calculateSMA = (data, window) => {
    return data.map((point, index, array) => {
      if (index < window - 1) return { ...point, [`sma${window}`]: null };
      const slice = array.slice(index - window + 1, index + 1);
      const sum = slice.reduce((acc, curr) => acc + curr.value, 0);
      return { ...point, [`sma${window}`]: sum / window };
    });
  };

  const dataWithSMA12 = React.useMemo(() => calculateSMA(wealthData, 12), [wealthData]);
  const finalData = React.useMemo(() => calculateSMA(dataWithSMA12, 60), [dataWithSMA12]);

  return (
    <div className="w-full bg-white p-6 rounded-sm border border-beige shadow-sm">
      <div className="flex justify-end gap-4 mb-4">
        <label className="flex items-center gap-2 text-xs font-bold uppercase text-slate-500 cursor-pointer hover:text-navy">
          <input type="checkbox" checked={showSMA12} onChange={e => setShowSMA12(e.target.checked)} className="accent-gold" />
          Show 1yr Avg
        </label>
        <label className="flex items-center gap-2 text-xs font-bold uppercase text-slate-500 cursor-pointer hover:text-navy">
          <input type="checkbox" checked={showSMA60} onChange={e => setShowSMA60(e.target.checked)} className="accent-[#4A7C59]" />
          Show 5yr Avg
        </label>
      </div>
      <div className="h-96 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={finalData}
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
            {showSMA12 && <Line type="monotone" dataKey="sma12" stroke="#D4AF37" strokeWidth={2} dot={false} name="1yr MA" />}
            {showSMA60 && <Line type="monotone" dataKey="sma60" stroke="#4A7C59" strokeWidth={2} dot={false} name="5yr MA" />}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default WealthChart;
