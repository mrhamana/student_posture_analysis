import React from 'react';
import { Clock, FileVideo, Image, CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import clsx from 'clsx';
import type { SessionData } from '../types';

interface SessionListProps {
  sessions: SessionData[];
  loading: boolean;
  currentSessionId: string | null;
  onSelect: (id: string) => void;
  onRefresh: () => void;
}

const statusConfig = {
  pending: { icon: Clock, color: 'text-amber-500', bg: 'bg-amber-50', label: 'Pending' },
  processing: { icon: Loader2, color: 'text-blue-500', bg: 'bg-blue-50', label: 'Processing' },
  completed: { icon: CheckCircle2, color: 'text-green-500', bg: 'bg-green-50', label: 'Completed' },
  failed: { icon: XCircle, color: 'text-red-500', bg: 'bg-red-50', label: 'Failed' },
};

export const SessionList: React.FC<SessionListProps> = ({
  sessions,
  loading,
  currentSessionId,
  onSelect,
  onRefresh,
}) => {
  return (
    <div className="card">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">Sessions</h3>
        <button
          className="btn-secondary text-xs"
          onClick={onRefresh}
          disabled={loading}
        >
          Refresh
        </button>
      </div>

      {sessions.length === 0 && !loading && (
        <p className="py-6 text-center text-sm text-gray-500">
          No sessions yet. Upload media to get started.
        </p>
      )}

      {loading && (
        <div className="flex items-center justify-center py-6">
          <Loader2 className="h-5 w-5 animate-spin text-gray-400" />
        </div>
      )}

      <div className="space-y-2 max-h-96 overflow-y-auto">
        {sessions.map((session) => {
          const config = statusConfig[session.status];
          const StatusIcon = config.icon;
          const isActive = session.id === currentSessionId;

          return (
            <button
              key={session.id}
              onClick={() => onSelect(session.id)}
              className={clsx(
                'flex w-full items-center gap-3 rounded-lg border px-3 py-2.5 text-left transition-all',
                isActive
                  ? 'border-primary-300 bg-primary-50 ring-1 ring-primary-200'
                  : 'border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50',
              )}
            >
              {session.media_type === 'video' ? (
                <FileVideo className="h-4 w-4 flex-shrink-0 text-gray-400" />
              ) : (
                <Image className="h-4 w-4 flex-shrink-0 text-gray-400" />
              )}

              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-gray-900">
                  {session.filename}
                </p>
                <p className="text-xs text-gray-500">
                  {new Date(session.created_at).toLocaleString()}
                </p>
              </div>

              <span
                className={clsx(
                  'badge flex items-center gap-1',
                  config.bg,
                  config.color,
                )}
              >
                <StatusIcon
                  className={clsx(
                    'h-3 w-3',
                    session.status === 'processing' && 'animate-spin',
                  )}
                />
                {config.label}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
};
