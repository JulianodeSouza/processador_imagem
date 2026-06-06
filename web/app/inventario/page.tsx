"use client";

import { ReactNode } from "react";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";

export default function InventarioPage(): ReactNode {
  return (
    <div className="w-full bg-surface">
      <Header title="BarberShop Atelier" />
      <Sidebar />

      <main className="lg:pl-72 pt-24 min-h-screen">
        <div className="px-12 py-8 max-w-7xl mx-auto">
          <section className="mb-12">
            <h1 className="text-4xl font-bold italic tracking-tight text-on-surface mb-2 font-headline">
              Inventário
            </h1>
            <p className="text-on-surface-variant font-label">
              Gerencie os produtos e ferramentas do seu atelier.
            </p>
          </section>

          <div className="glass-card rounded-xl p-8 border border-outline-variant/10">
            <p className="text-on-surface-variant">
              Gerenciamento de inventário será implementado aqui...
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
