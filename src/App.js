import React from "react";
import {
  BrowserRouter as Router,
  Route,
  Routes,
  useNavigate,
} from "react-router-dom";
import Header from "./components/Header/Header";
import Description from "./components/Description/Description";
import Menu from "./components/Menu/Menu";
import ScrollToExplore from "./components/ScrollToExplore/ScrollToExplore";
import Approach from "./components/Approach/Approach";
import VideoUpload from "./components/VideoUpload/VideoUpload";
import styles from "./App.module.css";
import Faq from "./components/Faq/Faq";
import DetailedApproach from "./components/DetailedApproach/DetailedApproach";

function App() {
  return (
    <Router>
      <div className={styles.app}>
        <Routes>
          <Route path="/" element={<MainContent />} />
          <Route path="/menu" element={<MenuContent />} />
          <Route path="/approach" element={<ApproachContent />} />
          <Route path="/faq" element={<FaqContent />} />
          <Route path="/demo" element={<DemoContent />} />
        </Routes>
      </div>
    </Router>
  );
}

function MainContent() {
  const navigate = useNavigate();
  const approachRef = React.useRef(null);

  const scrollToApproach = () => {
    approachRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const openMenu = () => {
    navigate("/menu");
  };

  return (
    <>
      <Header isMenuOpen={false} toggleMenu={openMenu} />
      <Description />
      <ScrollToExplore onClick={scrollToApproach} />
      <Approach ref={approachRef} />
      {/* <VideoUpload /> */}
    </>
  );
}

function MenuContent() {
  const navigate = useNavigate();

  const closeMenu = () => navigate("/");

  const openDemo = () => navigate("/demo");

  return (
    <>
      <Header isMenuOpen={true} toggleMenu={closeMenu} />
      <Menu onDemoClick={openDemo} />
    </>
  );
}

function DemoContent() {
  const navigate = useNavigate();

  const closeDemo = () => navigate("/menu");

  return (
    <>
      <Header isMenuOpen={false} toggleMenu={closeDemo} />
      <VideoUpload />
    </>
  );
}

function ApproachContent() {
  const navigate = useNavigate();

  const closeApproach = () => navigate("/menu");

  return (
    <>
      <Header isMenuOpen={false} toggleMenu={closeApproach} />
      <DetailedApproach />
    </>
  );
}

function FaqContent() {
  const navigate = useNavigate();

  const closeFaq = () => navigate("/menu");

  return (
    <>
      <Header isMenuOpen={false} toggleMenu={closeFaq} />
      <Faq />
    </>
  );
}

export default App;
