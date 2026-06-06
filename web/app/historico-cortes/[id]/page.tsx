"use client";

import { ReactNode, useState, useEffect } from "react";
import { useParams } from "next/navigation";
import { api } from "@/services/api";

interface HistoricoDetalhado {
  id: string;
  date: string;
  service: string;
  barberName: string;
  notes: string;
  photos: { id: string; url: string }[];
}

export default function HistoricoDetalhadoPage(): ReactNode {
  const { id } = useParams();
  const [data, setData] = useState<HistoricoDetalhado | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchHistorico() {
      try {
        // Rota conforme documento 
        const response = await api.get(`/historico-cortes/${id}`);
        setData(response.data);
      } catch (error) {
        console.error("Erro ao buscar histórico:", error);
      } finally {
        setIsLoading(false);
      }
    }
    if (id) fetchHistorico();
  }, [id]);

  if (isLoading) return <div className="p-12 text-on-surface-variant">Carregando detalhes...</div>;

  return (
    <div className="w-full bg-surface min-h-screen p-8 lg:p-12">
      <h1 className="text-3xl font-headline font-bold text-on-surface mb-8">
        Detalhes do Atendimento
      </h1>

      {data && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Informações do Atendimento */}
          <div className="glass-card p-6 rounded-2xl border border-outline-variant/20">
            <p className="text-primary font-label uppercase tracking-widest text-xs mb-2">
              {new Date(data.date).toLocaleDateString('pt-BR')}
            </p>
            <h2 className="text-2xl font-bold mb-4">{data.service}</h2>
            <p className="text-on-surface-variant mb-6">Barbeiro: {data.barberName}</p>
            <p className="text-sm bg-surface-container-high p-4 rounded-xl">{data.notes}</p>
          </div>

          {/* Galeria de Fotos */}
          <div className="glass-card p-6 rounded-2xl border border-outline-variant/20">
            <h3 className="font-bold mb-4">Fotos do Resultado</h3>
            <div className="grid grid-cols-2 gap-4">
              {data.photos.map((photo) => (
                <img 
                  key={photo.id} 
                  src={photo.url} 
                  alt="Resultado do corte" 
                  className="rounded-xl w-full h-40 object-cover border border-outline-variant/20"
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}