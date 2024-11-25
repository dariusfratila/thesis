import React, { forwardRef } from "react";
import styles from "./Approach.module.css";
import { useNavigate } from "react-router-dom";
import { ParallaxProvider, Parallax } from 'react-scroll-parallax';

const WaveSVG = () => (
  <svg
    className={styles.svgWave}
    viewBox="0 0 500 100"
    preserveAspectRatio="none"
  >
    <path
      d="M-0.00,39.98 C249.99,100.00 149.20,-19.98 600.00,59.98 L500.00,150.00 L0.00,150.00 Z"
      style={{ stroke: "none", fill: "white" }}
    ></path>
  </svg>
);

const Approach = forwardRef((props, ref) => {
  const navigate = useNavigate();

  const handleCardClick = () => {
    navigate("/approach");
  };

  return (
    <ParallaxProvider>
    <section ref={ref} className={styles.container}>
      <WaveSVG />
      <div className={styles.headingBox}>
        <h2 className={styles.heading} stroke="">
          Unlock the future of{" "}
          <span className={styles.lighterText}>lip-reading research</span>
        </h2>
        <Parallax className={styles.parallaxLine} x={[0, -70]}>
          <div className={styles.line}></div>
        </Parallax>
        <p className={styles.subtitle}>
          Discover the groundbreaking innovations in lip-reading technology,
          harnessing the power of lateral inhibition and the seamless
          integration of neural networks.
        </p>
      </div>
      <div className={styles.cardContainer}>
        {[
          "Data Augmentation for Enhanced Lip-Reading Accuracy",
          "Lateral Inhibition for Focus Enhancement",
          "Temporal Convolutional Network",
        ].map((title, index) => (
          <article
            key={index}
            className={styles.card}
            onClick={handleCardClick}
          >
            <div className={styles.cardContent}>
              <h3 className={styles.cardTitle}>{title}</h3>
              <p className={styles.cardDescription}>
                {index === 0 &&
                  "This approach employs various data augmentation techniques, such as random horizontal flips, mixup, and random cropping, to improve the robustness and accuracy of lip-reading models."}
                {index === 1 &&
                  "Inspired by neural processes, this method improves the network's focus on relevant signals, boosting performance in challenging conditions."}
                {index === 2 &&
                  "Uses TCNs to capture temporal features in lip movements."}
              </p>
            </div>
          </article>
        ))}
      </div>
    </section>
  </ParallaxProvider>
    
  );
});

export default Approach;
