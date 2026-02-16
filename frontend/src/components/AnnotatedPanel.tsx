import React, { useMemo, useEffect, useRef, useState } from 'react';
import type { PostureSummaryData, PostureClass, AnnotatedDetection } from '../types';
import { POSTURE_COLORS } from '../types';
import { getStudentProfile } from '../data/studentProfiles';
import { api } from '../api/client';

interface AnnotatedPanelProps {
    sessionId: string;
    sessionUpdatedAt?: string | null;
    summary: PostureSummaryData;
}

export const AnnotatedPanel: React.FC<AnnotatedPanelProps> = ({
    sessionId,
    sessionUpdatedAt,
    summary,
}) => {
    const students = useMemo(
        () => [...summary.students_summary].sort((a, b) => a.tracker_id - b.tracker_id),
        [summary.students_summary],
    );

    const cacheKey = sessionUpdatedAt ? `?t=${encodeURIComponent(sessionUpdatedAt)}` : '';
    const annotatedUrl = `/api/session/${sessionId}/annotated-image${cacheKey}`;

    const imageRef = useRef<HTMLImageElement | null>(null);
    const [detections, setDetections] = useState<AnnotatedDetection[]>([]);
    const [imageSize, setImageSize] = useState({
        naturalWidth: 0,
        naturalHeight: 0,
        displayWidth: 0,
        displayHeight: 0,
    });

    useEffect(() => {
        let active = true;
        api
            .getAnnotatedMetadata(sessionId)
            .then((data) => {
                if (active) setDetections(data.detections || []);
            })
            .catch(() => {
                if (active) setDetections([]);
            });
        return () => {
            active = false;
        };
    }, [sessionId]);

    useEffect(() => {
        const img = imageRef.current;
        if (!img) return;

        const updateSize = () => {
            setImageSize({
                naturalWidth: img.naturalWidth,
                naturalHeight: img.naturalHeight,
                displayWidth: img.clientWidth,
                displayHeight: img.clientHeight,
            });
        };

        updateSize();
        const observer = new ResizeObserver(updateSize);
        observer.observe(img);
        return () => observer.disconnect();
    }, [annotatedUrl]);

    const scaleX = imageSize.naturalWidth
        ? imageSize.displayWidth / imageSize.naturalWidth
        : 1;
    const scaleY = imageSize.naturalHeight
        ? imageSize.displayHeight / imageSize.naturalHeight
        : 1;

    return (
        <div className="card space-y-6">
            <div>
                <h3 className="text-lg font-semibold text-gray-900">Annotated Output</h3>
                <p className="text-xs text-gray-500">
                    Boxes and labels are drawn using the latest detected frame.
                </p>
            </div>

            <div className="overflow-hidden rounded-lg border border-gray-200 bg-gray-50">
                <div className="relative">
                    <img
                        ref={imageRef}
                        src={annotatedUrl}
                        alt="Annotated detection output"
                        className="h-auto w-full object-contain"
                        loading="lazy"
                        onLoad={() => {
                            const img = imageRef.current;
                            if (!img) return;
                            setImageSize({
                                naturalWidth: img.naturalWidth,
                                naturalHeight: img.naturalHeight,
                                displayWidth: img.clientWidth,
                                displayHeight: img.clientHeight,
                            });
                        }}
                    />

                    {detections.map((det) => {
                        const profile = getStudentProfile(det.tracker_id);
                        const postureColor =
                            POSTURE_COLORS[det.posture as PostureClass] || '#6b7280';
                        const [x1, y1] = det.bbox;
                        const left = x1 * scaleX + 4;
                        const top = y1 * scaleY + 4;
                        return (
                            <div
                                key={`${det.tracker_id}-${x1}-${y1}`}
                                className="absolute"
                                style={{ left, top }}
                            >
                                <div
                                    className="flex items-center gap-2 rounded-md bg-white/90 px-2 py-1 shadow"
                                    style={{ border: `1px solid ${postureColor}` }}
                                >
                                    <img
                                        src={profile.imageUrl}
                                        alt={profile.name}
                                        className="h-7 w-7 rounded object-cover"
                                        loading="lazy"
                                    />
                                    <span className="text-xs font-semibold text-gray-900">
                                        ID {det.tracker_id}
                                    </span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            <div>
                <div className="mb-3 flex items-center justify-between">
                    <h4 className="text-sm font-semibold text-gray-900">Detected Students</h4>
                    <span className="text-xs text-gray-500">{students.length} total</span>
                </div>

                {students.length === 0 ? (
                    <p className="py-4 text-center text-sm text-gray-500">No students detected</p>
                ) : (
                    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                        {students.map((student) => {
                            const profile = getStudentProfile(student.tracker_id);
                            const postureColor =
                                POSTURE_COLORS[student.current_posture as PostureClass] || '#6b7280';
                            const cropUrl = `/api/session/${sessionId}/students/${student.tracker_id}/crop?t=${student.last_seen_frame}`;

                            return (
                                <div
                                    key={student.tracker_id}
                                    className="flex items-center gap-3 rounded-lg border border-gray-200 bg-white p-3"
                                >
                                    <img
                                        src={cropUrl}
                                        alt={`${profile.name} crop`}
                                        className="h-16 w-16 shrink-0 rounded-lg object-cover"
                                        onError={(event) => {
                                            (event.currentTarget as HTMLImageElement).src = profile.imageUrl;
                                        }}
                                        loading="lazy"
                                    />

                                    <div className="min-w-0 flex-1">
                                        <p className="truncate text-sm font-semibold text-gray-900">
                                            {profile.name}
                                        </p>
                                        <p className="text-xs text-gray-500">Assigned ID: #{profile.trackerId}</p>
                                        <div className="mt-2 flex items-center gap-2">
                                            <span
                                                className="badge"
                                                style={{
                                                    backgroundColor: `${postureColor}20`,
                                                    color: postureColor,
                                                }}
                                            >
                                                {student.current_posture || 'N/A'}
                                            </span>
                                            <span className="text-xs text-gray-500">
                                                {(student.current_confidence * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>
        </div>
    );
};
