import { redirect } from 'next/navigation'

// Always redirect to /login — the login page redirects to /library if already authenticated.
// Server-side redirect: no blank flash, no client JS required.
export default function RootPage() {
  redirect('/login')
}
