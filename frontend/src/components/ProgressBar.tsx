import React from 'react';
import { Loader2, CheckCircle2, XCircle, Clock } from 'lucide-react';
import type { ProcessingProgress, SessionData } from '../types';

interface ProgressBarProps {
  progress: ProcessingProgress | null;
  session: SessionData | null;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ progress, session }) => {
  if (!session) return null;

  const status = session.status;
  const percent = progress?.progress_percent ?? 0;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {status === 'processing' && (
            <Loader2 className="h-5 w-5 animate-spin text-primary-500" />
          )}
          {status === 'pending' && <Clock className="h-5 w-5 text-amber-500" />}
          {status === 'completed' && <CheckCircle2 className="h-5 w-5 text-green-500" />}
          {status === 'failed' && <XCircle className="h-5 w-5 text-red-500" />}

          <span className="text-sm font-medium text-gray-700">
            {status === 'pending' && 'Waiting to process...'}
            {status === 'processing' && 'Analyzing postures...'}
            {status === 'completed' && 'Analysis complete'}
            {status === 'failed' && 'Analysis failed'}
          </span>
        </div>

        {status === 'processing' && progress && (
          <span className="text-sm font-semibold text-primary-600">
            {progress.processed_frames} / {progress.total_frames || '?'} frames
          </span>
        )}
      </div>

      {(status === 'processing' || status === 'pending') && (
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-gray-200">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              status === 'pending'
                ? 'animate-pulse bg-amber-400'
                : 'bg-primary-500'
            }`}
            style={{ width: status === 'pending' ? '100%' : `${percent}%` }}
          />
        </div>
      )}

      {status === 'failed' && session.error_message && (
        <div className="mt-2 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700">
          {session.error_message}
        </div>
      )}
    </div>
  );
};
