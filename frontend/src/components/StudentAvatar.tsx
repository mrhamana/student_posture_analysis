import React, { useEffect, useState } from 'react';
import type { StudentProfile } from '../data/studentProfiles';

const FALLBACK_IMAGE = '/students/placeholder.svg';

interface StudentAvatarProps {
    profile: StudentProfile;
    sizeClassName?: string;
    className?: string;
}

export const StudentAvatar: React.FC<StudentAvatarProps> = ({
    profile,
    sizeClassName = 'h-10 w-10',
    className = '',
}) => {
    const [src, setSrc] = useState(profile.imageUrl);

    useEffect(() => {
        setSrc(profile.imageUrl);
    }, [profile.imageUrl]);

    return (
        <img
            src={src}
            alt={profile.name}
            className={`shrink-0 rounded-full object-cover ${sizeClassName} ${className}`}
            onError={() => setSrc(FALLBACK_IMAGE)}
            loading="lazy"
        />
    );
};
