import java.sql.Driver;
import java.time.Duration;


import org.openqa.selenium.By;
import org.openqa.selenium.TimeoutException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

public class Sc2 {
	public void run2() {
		
		WebDriver driver = new ChromeDriver();
		
		try {
			
			
			driver.get("https://the-internet.herokuapp.com/dynamic_loading/2");
			driver.manage().window().maximize();
			
			driver.findElement(By.xpath("//*[@id=\"start\"]/button")).click();
			
			// Wait 30 + 1 seconds
			
			WebDriverWait wait31sec = new WebDriverWait(driver, Duration.ofSeconds(31));
			
			wait31sec.until(ExpectedConditions.visibilityOfElementLocated(By.xpath("//*[@id=\"finish\"]")));
			
			String result = driver.findElement(By.xpath("//*[@id=\"finish\"]")).getText();
			
			// Validation
			
			if (result.isEmpty()) {
				System.out.println("Error");
			} else {
				System.out.println("Isi Konten" + result);
			}
			
			
		} catch (TimeoutException e) {
			// TODO Auto-generated catch block
			System.out.println("Timedout: " + e.getMessage());
			
		} finally {
			driver.close();
		}
		
		
	}
}
