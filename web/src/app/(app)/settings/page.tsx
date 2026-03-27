import ApiKeySettings from '@/components/settings/ApiKeySettings'

export const metadata = {
  title: 'Settings — Sunday Album',
}

export default function SettingsPage() {
  return (
    <div>
      <h1 className="font-display text-2xl font-bold text-sa-stone-900 dark:text-sa-stone-50 mb-6">
        Settings
      </h1>
      <ApiKeySettings />
    </div>
  )
}
