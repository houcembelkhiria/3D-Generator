import React from 'react';

interface StatusBadgeProps {
  label: string;
  status: 'online' | 'offline' | 'busy';
  value?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ label, status, value }) => {
  const getStatusColor = () => {
    switch (status) {
      case 'online': return 'bg-[#7C3AED]/10 text-[#7C3AED] border-[#7C3AED]/20';
      case 'offline': return 'bg-red-500/10 text-red-500 border-red-500/20';
      case 'busy': return 'bg-[#FF8C66]/10 text-[#FF8C66] border-[#FF8C66]/20';
      default: return 'bg-zinc-500/10 text-zinc-400 border-zinc-500/20';
    }
  };

  const getDotColor = () => {
    switch (status) {
      case 'online': return 'bg-[#7C3AED]';
      case 'offline': return 'bg-red-500';
      case 'busy': return 'bg-[#FF8C66]';
      default: return 'bg-zinc-500';
    }
  };

  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-medium ${getStatusColor()}`}>
      <span className={`w-2 h-2 rounded-full ${getDotColor()} animate-pulse`} />
      <span className="uppercase tracking-wide">{label}</span>
      {value && <span className="font-bold border-l border-current pl-2 ml-1">{value}</span>}
    </div>
  );
};