declare module "react-syntax-highlighter" {
  import { ComponentType } from "react";
  export const Prism: ComponentType<any>;
  export default ComponentType;
}

declare module "react-syntax-highlighter/dist/esm/styles/prism" {
  const styles: Record<string, Record<string, React.CSSProperties>>;
  export const oneDark: Record<string, React.CSSProperties>;
  export const oneLight: Record<string, React.CSSProperties>;
  export default styles;
}
