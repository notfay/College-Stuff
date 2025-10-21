import java.time.Duration;
import java.util.NoSuchElementException;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

public class Sc1 {
	public void run() {
	
		WebDriver driver = new ChromeDriver();
		driver.get("https://the-internet.herokuapp.com/login");
		
		driver.findElement(By.xpath("//*[@id=\"username\"]")).sendKeys("tomsmith");
		
		driver.findElement(By.xpath("//*[@id=\"password\"]")).sendKeys("SuperSecretPassword!");
		
		driver.findElement(By.xpath("//*[@id=\"login\"]/button/i")).click();
		
		WebDriverWait wait5sec = new WebDriverWait(driver, Duration.ofSeconds(5));
		
		// Explicit 5 sec wait
		
		try {
			wait5sec.until(ExpectedConditions.visibilityOfElementLocated(By.xpath("//*[@id=\"content\"]/div/h2/text()")));
			
			System.out.println("Login Secured!");
			
		} catch (NoSuchElementException e) {
			System.out.println("Failed! " + e.getMessage());
		}
		
		try {
			Thread.sleep(3);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		driver.close();
		
		
		
	}
}
