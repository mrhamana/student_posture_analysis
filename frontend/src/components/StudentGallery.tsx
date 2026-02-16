import React, { useMemo } from 'react';
import type { PostureSummaryData, PostureClass } from '../types';
import { POSTURE_COLORS } from '../types';
import { getStudentProfile } from '../data/studentProfiles';
import { StudentAvatar } from './StudentAvatar';

interface StudentGalleryProps {
    summary: PostureSummaryData;
}

export const StudentGallery: React.FC<StudentGalleryProps> = ({ summary }) => {
    const students = useMemo(
        () => [...summary.students_summary].sort((a, b) => a.tracker_id - b.tracker_id),
        [summary.students_summary],
    );

    return (
        <div className="card">
            <div className="mb-4 flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-900">Students</h3>
                <span className="text-xs text-gray-500">{students.length} total</span>
            </div>

            {students.length === 0 ? (
                <p className="py-6 text-center text-sm text-gray-500">No student data available</p>
            ) : (
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                    {students.map((student) => {
                        const profile = getStudentProfile(student.tracker_id);
                        const postureColor =
                            POSTURE_COLORS[student.current_posture as PostureClass] || '#6b7280';

                        return (
                            <div
                                key={student.tracker_id}
                                className="group flex items-center gap-3 rounded-lg border border-gray-200 bg-white p-3 transition-shadow hover:shadow-sm"
                            >
                                <StudentAvatar profile={profile} sizeClassName="h-12 w-12" />

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
    );
};
