import React, { useEffect, useCallback, useMemo } from 'react';
import { MonitorCheck } from 'lucide-react';
import { useAppState } from './hooks/useAppState';
import { FileUpload } from './components/FileUpload';
import { DetectionCanvas } from './components/DetectionCanvas';
import { PostureTable } from './components/PostureTable';
import { SessionSummary } from './components/SessionSummary';
import { ProgressBar } from './components/ProgressBar';
import { SessionList } from './components/SessionList';
import type { DetectionOverlay, PostureClass } from './types';

const App: React.FC = () => {
  const {
    state,
    uploadFile,
    pollProgress,
    selectSession,
    loadSessions,
  } = useAppState();

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // Start polling after upload
  useEffect(() => {
    if (
      state.currentSessionId &&
      state.session &&
      (state.session.status === 'pending' || state.session.status === 'processing')
    ) {
      pollProgress(state.currentSessionId);
    }
  }, [state.currentSessionId, state.session?.status]);

  // Handle upload
  const handleUpload = useCallback(
    async (file: File) => {
      const sessionId = await uploadFile(file);
      if (sessionId) {
        loadSessions();
        selectSession(sessionId);
      }
      return sessionId;
    },
    [uploadFile, loadSessions, selectSession],
  );

  // Build detection overlays from records
  const detections: DetectionOverlay[] = useMemo(() => {
    if (!state.records.length) return [];
    return state.records.map((r) => {
      // Find tracker_id from students list
      const student = state.students.find((s) => s.id === r.student_id);
      return {
        trackerId: student?.tracker_id ?? r.student_id,
        bbox: [r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2] as [number, number, number, number],
        posture: r.posture as PostureClass,
        confidence: r.confidence,
      };
    });
  }, [state.records, state.students]);

  const isCompleted = state.session?.status === 'completed';
  const isProcessing =
    state.session?.status === 'processing' || state.session?.status === 'pending';

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-gray-200 bg-white/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-6 py-4">
          <div className="rounded-lg bg-primary-600 p-2">
            <MonitorCheck className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-900">Student Posture Analysis</h1>
            <p className="text-xs text-gray-500">AI-powered CCTV posture detection & tracking</p>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl p-6">
        <div className="grid gap-6 lg:grid-cols-[300px_1fr]">
          {/* Sidebar */}
          <aside className="space-y-6">
            <SessionList
              sessions={state.sessions}
              loading={state.sessionsLoading}
              currentSessionId={state.currentSessionId}
              onSelect={selectSession}
              onRefresh={loadSessions}
            />
          </aside>

          {/* Main Content */}
          <div className="space-y-6">
            {/* Upload Section */}
            <FileUpload
              onUpload={handleUpload}
              uploading={state.uploading}
              uploadProgress={state.uploadProgress}
              uploadError={state.uploadError}
            />

            {/* Progress */}
            {state.currentSessionId && (isProcessing || state.session?.status === 'failed') && (
              <ProgressBar progress={state.processingProgress} session={state.session} />
            )}

            {/* Results */}
            {isCompleted && state.summary && (
              <>
                {/* Summary */}
                <SessionSummary summary={state.summary} />

                {/* Visualization */}
                {state.session?.file_path && (
                  <div className="card">
                    <h3 className="mb-4 text-lg font-semibold text-gray-900">
                      Detection Visualization
                    </h3>
                    <DetectionCanvas
                      mediaUrl={state.session.file_path}
                      mediaType={state.session.media_type}
                      detections={detections}
                    />
                  </div>
                )}

                {/* Table */}
                <PostureTable summary={state.summary} />
              </>
            )}

            {/* Empty state */}
            {!state.currentSessionId && !state.uploading && (
              <div className="card flex flex-col items-center justify-center py-16 text-center">
                <MonitorCheck className="mb-4 h-12 w-12 text-gray-300" />
                <h3 className="mb-1 text-lg font-medium text-gray-600">
                  No session selected
                </h3>
                <p className="text-sm text-gray-400">
                  Upload an image or video, or select a previous session from the sidebar.
                </p>
              </div>
            )}

            {/* Loading results */}
            {state.resultsLoading && (
              <div className="card flex items-center justify-center py-12">
                <div className="flex items-center gap-3 text-gray-500">
                  <div className="h-5 w-5 animate-spin rounded-full border-2 border-gray-300 border-t-primary-500" />
                  <span className="text-sm">Loading results...</span>
                </div>
              </div>
            )}

            {/* Results error */}
            {state.resultsError && (
              <div className="card border-red-200 bg-red-50">
                <p className="text-sm text-red-700">{state.resultsError}</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
