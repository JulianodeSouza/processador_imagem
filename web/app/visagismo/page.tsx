"use client";

import { ReactNode, useState } from "react";
import ImageUploadModule, { AnalysisResult } from "@/components/ImageUploadModule";

interface AnaliseVisagismo {
  id: string;
  clientName: string;
  date: string;
  result: string;
}

export default function VisagismoPage(): ReactNode {
  const [analysisHistory, setAnalysisHistory] = useState<AnaliseVisagismo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [currentAnalysis, setCurrentAnalysis] = useState<AnalysisResult | null>(null);

  return (
    <div className="w-full bg-surface min-h-screen">
      <main className="lg:pl-72 pt-28 pb-20">
        <div className="px-6 lg:px-12 max-w-7xl mx-auto">

          {/* Header */}
          <section className="mb-12">
            <h1 className="text-4xl lg:text-5xl font-headline font-bold text-on-surface mb-3 tracking-wide">
              Visagismo IA
            </h1>
            <p className="text-on-surface-variant font-label text-sm uppercase tracking-wider">
              Análise facial inteligente e recomendação de cortes personalizados.
            </p>
          </section>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

            {/* Coluna esquerda: Upload + Resultado */}
            <div className="lg:col-span-1 space-y-6">
              <div className="glass-card rounded-2xl p-6 border border-outline-variant/20">
                <h2 className="font-headline text-lg font-bold mb-6">
                  Nova Análise
                </h2>
                <ImageUploadModule
                  clientId="cliente-atual-id"
                  onAnalysisComplete={setCurrentAnalysis}
                />
              </div>

              {/* Resultado da análise */}
              {currentAnalysis && (
                <div className="glass-card rounded-2xl p-6 border border-primary/50 bg-primary/5 animate-in fade-in slide-in-from-bottom-4">
                  <h3 className="font-headline font-bold text-primary mb-4 flex items-center gap-2">
                    <span className="material-symbols-outlined">auto_awesome</span>
                    Resultado da IA
                  </h3>

                  <div className="space-y-4 text-sm">

                    {/* Formato do Rosto */}
                    <div>
                      <p className="text-on-surface-variant font-label text-xs uppercase tracking-wider mb-1">
                        Formato do Rosto
                      </p>
                      <p className="font-bold text-on-surface">
                        {currentAnalysis?.recommendation?.faceShape}
                      </p>
                    </div>

                    {/* Confiança */}
                    <div>
                      <p className="text-on-surface-variant font-label text-xs uppercase tracking-wider mb-1">
                        Confiança da IA
                      </p>
                      <p className="font-bold text-primary">
                       {currentAnalysis?.recommendation?.confidence 
  ? Math.round(currentAnalysis.recommendation.confidence * 100) 
  : 0}%
                      </p>
                    </div>

                    {/* Descrição */}
                    <div>
                      <p className="text-on-surface-variant font-label text-xs uppercase tracking-wider mb-1">
                        Descrição
                      </p>
                      <p className="text-on-surface leading-relaxed">
                        {currentAnalysis.recommendation.description}
                      </p>
                    </div>

                    {/* Justificativa */}
                    <div>
                      <p className="text-on-surface-variant font-label text-xs uppercase tracking-wider mb-1">
                        Dica Principal
                      </p>
                      <p className="text-on-surface leading-relaxed">
                        {currentAnalysis.recommendation.justification}
                      </p>
                    </div>

                    {/* Cortes Sugeridos */}
                    <div>
                      <p className="text-on-surface-variant font-label text-xs uppercase tracking-wider mb-2">
                        Cortes Sugeridos
                      </p>
                      <div className="space-y-2">
                        {currentAnalysis.recommendation.suggestedCuts.map((corte, idx) => (
                          <div
                            key={idx}
                            className="p-3 bg-surface-container-high rounded-xl border border-outline-variant/30"
                          >
                            <p className="font-bold text-on-surface text-xs">{corte.nome}</p>
                            <p className="text-on-surface-variant text-xs mt-0.5">{corte.justificativa}</p>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Evitar */}
                    {currentAnalysis.recommendation.avoid?.length > 0 && (
                      <div>
                        <p className="text-on-surface-variant font-label text-xs uppercase tracking-wider mb-2">
                          Evitar
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {currentAnalysis.recommendation.avoid.map((item, idx) => (
                            <span
                              key={idx}
                              className="px-3 py-1 bg-error/10 text-error rounded-full text-xs font-bold border border-error/20"
                            >
                              {item}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Foto enviada */}
                    {currentAnalysis.photoUrl && (
                      <div>
                        <p className="text-on-surface-variant font-label text-xs uppercase tracking-wider mb-2">
                          Foto Analisada
                        </p>
                        <img
                          src={currentAnalysis.photoUrl}
                          alt="Foto analisada"
                          className="w-full rounded-xl object-cover border border-outline-variant/20"
                        />
                      </div>
                    )}

                  </div>
                </div>
              )}
            </div>

            {/* Coluna direita: Histórico */}
            <div className="lg:col-span-2">
              <div className="glass-card rounded-2xl p-8 border border-outline-variant/20">
                <h2 className="font-headline text-lg font-bold mb-6">
                  Histórico de Sugestões
                </h2>
                <div className="space-y-4">
                  {isLoading ? (
                    <p className="text-on-surface-variant text-sm animate-pulse">
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
                          <p className="font-bold text-on-surface">{item.clientName}</p>
                          <p className="text-xs text-on-surface-variant">{item.date}</p>
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
      </main>
    </div>
  );
}
