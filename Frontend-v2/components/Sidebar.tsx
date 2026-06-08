import React from 'react';
import { AppView } from '../types';
import { IconLayoutDashboard, IconSettings, IconFileText, IconUpload, IconMessageSquare, IconBox, IconDatabase } from './Icons';

interface SidebarProps {
  activeView: AppView;
  onViewChange: (view: AppView) => void;
}

const NAV_ITEMS: { view: AppView; label: string; icon: React.ElementType; section?: string }[] = [
  { view: 'agent', label: 'Agent', icon: IconLayoutDashboard },
  { view: 'files', label: 'Files Treatment', icon: IconFileText },
  { section: '3D Generation', view: 'image-to-3d', label: 'Image to 3D', icon: IconUpload },
  { view: 'text-to-3d', label: 'Text to 3D', icon: IconMessageSquare },
  { view: 'multiview-to-3d', label: 'Multi-View', icon: IconBox },
  { view: 'gallery', label: 'Gallery', icon: IconDatabase },
  { section: 'System', view: 'settings', label: 'System Config', icon: IconSettings },
];

export const Sidebar: React.FC<SidebarProps> = ({ activeView, onViewChange }) => {
  return (
    <div className="w-20 lg:w-64 sidebar flex flex-col items-center lg:items-stretch py-6 space-y-1 flex-shrink-0 overflow-y-auto">
      <div className="flex items-center justify-center lg:justify-start lg:px-6 mb-6">
        <div className="w-8 h-8 bg-gradient-to-br from-[#FF8C66] to-[#7C3AED] rounded-lg flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-[#FF8C66]/20">
          M
        </div>
        <span className="hidden lg:block ml-3 font-bold text-lg text-theme-primary tracking-tight">MCP Agent</span>
      </div>

      {NAV_ITEMS.map((item, i) => (
        <React.Fragment key={item.view}>
          {item.section && (
            <div className="hidden lg:block px-6 pt-4 pb-1">
              <span className="text-[10px] font-semibold text-theme-muted uppercase tracking-widest">{item.section}</span>
            </div>
          )}
          <button
            onClick={() => onViewChange(item.view)}
            className={`w-full lg:w-auto mx-2 lg:mx-4 p-3 lg:px-4 lg:py-2.5 flex items-center justify-center lg:justify-start sidebar-item ${
              activeView === item.view ? 'active' : ''
            }`}
          >
            <item.icon className={`w-5 h-5 ${activeView === item.view ? 'text-white' : 'text-theme-muted'}`} />
            <span className="hidden lg:block ml-3 text-sm font-medium text-theme-primary">{item.label}</span>
          </button>
        </React.Fragment>
      ))}
    </div>
  );
};
