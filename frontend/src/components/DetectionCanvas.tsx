import React, { useRef, useEffect, useCallback } from 'react';
import type { DetectionOverlay, PostureClass } from '../types';
import { POSTURE_COLORS } from '../types';

interface DetectionCanvasProps {
  mediaUrl: string;
  mediaType: 'image' | 'video';
  detections: DetectionOverlay[];
  width?: number;
  height?: number;
}

export const DetectionCanvas: React.FC<DetectionCanvasProps> = ({
  mediaUrl,
  mediaType,
  detections,
  width = 1280,
  height = 720,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaRef = useRef<HTMLImageElement | HTMLVideoElement | null>(null);
  const animFrameRef = useRef<number>(0);

  const drawOverlays = useCallback(
    (ctx: CanvasRenderingContext2D, displayWidth: number, displayHeight: number) => {
      for (const det of detections) {
        const color = POSTURE_COLORS[det.posture as PostureClass] || '#ffffff';
        const [x1, y1, x2, y2] = det.bbox;
        const bw = x2 - x1;
        const bh = y2 - y1;

        // Bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, bw, bh);

        // Label background
        const label = `#${det.trackerId} ${det.posture} (${(det.confidence * 100).toFixed(0)}%)`;
        ctx.font = 'bold 13px Inter, system-ui, sans-serif';
        const textMetrics = ctx.measureText(label);
        const textHeight = 18;
        const padding = 4;

        const labelY = y1 - textHeight - padding * 2;
        const labelBgY = labelY < 0 ? y1 : labelY;

        ctx.fillStyle = color;
        ctx.globalAlpha = 0.85;
        ctx.fillRect(
          x1,
          labelBgY,
          textMetrics.width + padding * 2,
          textHeight + padding,
        );
        ctx.globalAlpha = 1;

        // Label text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, x1 + padding, labelBgY + textHeight);
      }
    },
    [detections],
  );

  const renderFrame = useCallback(() => {
    const canvas = canvasRef.current;
    const media = mediaRef.current;
    if (!canvas || !media) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (media instanceof HTMLVideoElement) {
      ctx.drawImage(media, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.drawImage(media, 0, 0, canvas.width, canvas.height);
    }

    drawOverlays(ctx, canvas.width, canvas.height);

    if (media instanceof HTMLVideoElement && !media.paused && !media.ended) {
      animFrameRef.current = requestAnimationFrame(renderFrame);
    }
  }, [drawOverlays]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (mediaType === 'image') {
      const img = new window.Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        canvas.width = img.naturalWidth || width;
        canvas.height = img.naturalHeight || height;
        mediaRef.current = img;
        renderFrame();
      };
      img.src = mediaUrl;
    } else {
      const video = document.createElement('video');
      video.crossOrigin = 'anonymous';
      video.src = mediaUrl;
      video.muted = true;
      video.playsInline = true;

      video.addEventListener('loadeddata', () => {
        canvas.width = video.videoWidth || width;
        canvas.height = video.videoHeight || height;
        mediaRef.current = video;
        renderFrame();
      });

      video.addEventListener('play', () => {
        animFrameRef.current = requestAnimationFrame(renderFrame);
      });

      video.load();

      return () => {
        video.pause();
        video.src = '';
        if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      };
    }
  }, [mediaUrl, mediaType, width, height, renderFrame]);

  // Re-draw overlays when detections change (for images)
  useEffect(() => {
    if (mediaType === 'image' && mediaRef.current) {
      renderFrame();
    }
  }, [detections, mediaType, renderFrame]);

  return (
    <div className="relative w-full overflow-hidden rounded-lg border border-gray-200 bg-black">
      <canvas
        ref={canvasRef}
        className="w-full h-auto"
        style={{ maxHeight: '70vh' }}
      />
      {mediaType === 'video' && mediaRef.current && (
        <div className="absolute bottom-3 left-3 flex gap-2">
          <button
            className="rounded-md bg-black/60 px-3 py-1.5 text-xs font-medium text-white backdrop-blur-sm hover:bg-black/80"
            onClick={() => {
              const video = mediaRef.current as HTMLVideoElement;
              if (video.paused) {
                video.play();
              } else {
                video.pause();
                renderFrame();
              }
            }}
          >
            Play / Pause
          </button>
        </div>
      )}

      {detections.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <p className="rounded-lg bg-black/50 px-4 py-2 text-sm text-white">
            No detections to display
          </p>
        </div>
      )}
    </div>
  );
};
