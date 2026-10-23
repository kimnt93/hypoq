import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'vnresearch AutoResearch Lab — Full Stack Quantitative Hub',
  description: 'Autonomous Quantitative Research Harness for Vietnamese Financial Markets',
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
