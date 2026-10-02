export const hasAdminNavigation = (user, userId) => Boolean(userId && user?.id === userId && Array.isArray(user.roles) && user.roles.includes('admin'));
