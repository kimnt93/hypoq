import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'AI AutoResearch Lab',
  description: 'Autonomous Quantitative Research Network',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
