import AuthForm from '@/components/auth/AuthForm'

export const metadata = {
  title: 'Sign In — Sunday Album',
}

export default function LoginPage() {
  return (
    <main className="min-h-screen flex items-center justify-center px-4 bg-sa-stone-50 dark:bg-sa-stone-950">
      <div className="w-full max-w-md">
        <div className="mb-10 text-center">
          <h1 className="font-display text-4xl font-bold text-sa-stone-900 dark:text-sa-stone-50 mb-2">
            Sunday Album
          </h1>
          <p className="text-sa-stone-500 dark:text-sa-stone-400">
            Digitise your physical photo albums
          </p>
        </div>
        <AuthForm />
      </div>
    </main>
  )
}
