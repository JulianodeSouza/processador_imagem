"use client";

import { useState, ReactNode } from "react";
import Header from "./Header";
import Sidebar from "./Sidebar";

interface LayoutWrapperProps {
  children: ReactNode;
}

export default function LayoutWrapper({
  children,
}: LayoutWrapperProps): ReactNode {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const closeSidebar = () => {
    setSidebarOpen(false);
  };

  return (
    <div className="w-full bg-surface min-h-screen">
      <Header onSidebarToggle={toggleSidebar} />
      <Sidebar isOpen={sidebarOpen} onClose={closeSidebar} />
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          onClick={closeSidebar}
        />
      )}
      {children}
    </div>
  );
}
