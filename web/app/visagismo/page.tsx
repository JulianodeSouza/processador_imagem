"use client";

import { ReactNode, useState } from "react";
import ImageUploadModule, { AnalysisResult } from "@/components/ImageUploadModule"; // Importe a interface

interface AnaliseVisagismo {
  id: string;
  clientName: string;
  date: string;
  result: string;
}

export default function VisagismoPage(): ReactNode {
  const [analysisHistory, setAnalysisHistory] = useState<AnaliseVisagismo[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // 1. Novo estado para guardar o resultado da análise atual
  const [currentAnalysis, setCurrentAnalysis] = useState<AnalysisResult | null>(null);

  // ... (useEffect do histórico intacto) ...

  return (
    <div className="w-full bg-surface min-h-screen">
      {/* ... header e sidebar ... */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Upload e Análise */}
        <div className="lg:col-span-1">
          <div className="glass-card rounded-2xl p-6 border border-outline-variant/20">
            <h2 className="font-headline text-lg font-bold mb-6">
              Nova Análise
            </h2>

            {/* 2. Passamos a função atualizadora para o onAnalysisComplete */}
            <ImageUploadModule
              clientId="cliente-atual-id"
              onAnalysisComplete={setCurrentAnalysis}
            />
          </div>

          {/* 3. A Caixa de Resultado renderizada condicionalmente */}
          {currentAnalysis && (
            <div className="mt-6 glass-card rounded-2xl p-6 border border-primary/50 bg-primary/5 animate-in fade-in slide-in-from-bottom-4">
              <h3 className="font-headline font-bold text-primary mb-4 flex items-center gap-2">
                <span className="material-symbols-outlined">auto_awesome</span>
                Resultado da IA
              </h3>

              <div className="space-y-4 text-sm">
                <div>
                  <p className="text-on-surface-variant font-label text-xs uppercase tracking-wider mb-1">Formato do Rosto</p>
                  <p className="font-bold text-on-surface">{currentAnalysis.recommendation.faceShape}</p>
                </div>

                <div>
                  <p className="text-on-surface-variant font-label text-xs uppercase tracking-wider mb-1">Justificativa</p>
                  <p className="text-on-surface leading-relaxed">{currentAnalysis.recommendation.justification}</p>
                </div>

                <div>
                  <p className="text-on-surface-variant font-label text-xs uppercase tracking-wider mb-2">Cortes Sugeridos</p>
                  <div className="flex flex-wrap gap-2">
                    {currentAnalysis.recommendation.suggestedCuts.map((corte, idx) => (
                      <span key={idx} className="px-3 py-1 bg-surface-container-high text-on-surface rounded-full text-xs font-bold border border-outline-variant/30">
                        {corte}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Histórico de sugestões */}
        <div className="lg:col-span-2">
          <div className="glass-card rounded-2xl p-8 border border-outline-variant/20">
            <h2 className="font-headline text-lg font-bold mb-6">
              Histórico de Sugestões
            </h2>
            <div className="space-y-4">
              {isLoading ? (
                <p className="text-on-surface-variant text-sm">
                  Carregando histórico...
                </p>
              ) : analysisHistory.length === 0 ? (
                <p className="text-on-surface-variant text-sm">
                  Nenhuma análise realizada ainda.
                </p>
              ) : (
                analysisHistory.map((item) => (
                  <div
                    key={item.id}
                    className="flex items-center justify-between p-4 bg-surface-container-low rounded-xl border border-outline-variant/10 transition-all hover:bg-surface-container"
                  >
                    <div>
                      <p className="font-bold text-on-surface">
                        {item.clientName}
                      </p>
                      <p className="text-xs text-on-surface-variant">
                        {item.date}
                      </p>
                    </div>
                    <span className="px-3 py-1 bg-primary/10 text-primary rounded-full font-label text-xs font-bold border border-primary/20">
                      {item.result}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
